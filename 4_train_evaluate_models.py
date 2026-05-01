"""
Fine-tunes FinBERT, RoBERTa-base and DistilBERT on the final dataset,
selects the best model via validation F1-macro, and reports test performance
with confusion matrices.

Requirements:
    pip install pandas scikit-learn transformers torch accelerate matplotlib seaborn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE   = "sp500_final_dataset.csv"
OUTPUT_DIR   = "model_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    "FinBERT"    : "ProsusAI/finbert",
    "RoBERTa"    : "roberta-base",
    "DistilBERT" : "distilbert-base-uncased",
}

# Label mapping: var_class (-1, 0, 1) → model label index (0, 1, 2)
LABEL2IDX = {-1: 0, 0: 1, 1: 2}
IDX2LABEL = {0: -1, 1: 0, 2: 1}
CLASS_NAMES = ["DOWN", "FLAT", "UP"]

# Training hyperparameters
MAX_LEN    = 128
BATCH_SIZE = 16
EPOCHS     = 3
LR         = 2e-5
VAL_RATIO  = 0.15
TEST_RATIO = 0.15
SEED       = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
# ─────────────────────────────────────────────────────────────────────────────


# ── Dataset ───────────────────────────────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids"      : enc["input_ids"].squeeze(0),
            "attention_mask" : enc["attention_mask"].squeeze(0),
            "labels"         : torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(
            input_ids      = batch["input_ids"].to(DEVICE),
            attention_mask = batch["attention_mask"].to(DEVICE),
            labels         = batch["labels"].to(DEVICE),
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += out.loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids      = batch["input_ids"].to(DEVICE),
                attention_mask = batch["attention_mask"].to(DEVICE),
            )
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())
    return np.array(all_preds), np.array(all_labels)


# ── Confusion matrix plot ─────────────────────────────────────────────────────
def plot_confusion_matrix(cm, model_name, split_name, ax):
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(f"{model_name}\n({split_name})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["titles", "var_class"])
    df["label"] = df["var_class"].astype(int).map(LABEL2IDX)
    print(f"Total samples: {len(df)}")
    print("Class distribution:\n", df["var_class"].value_counts().sort_index())

    texts  = df["titles"].tolist()
    labels = df["label"].tolist()

    # ── Stratified split: train / val / test ──────────────────────────────────
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        texts, labels,
        test_size=VAL_RATIO + TEST_RATIO,
        stratify=labels,
        random_state=SEED,
    )
    val_size_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=1 - val_size_adjusted,
        stratify=y_tmp,
        random_state=SEED,
    )
    print(f"\nSplit — train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}")

    # ── Results storage ───────────────────────────────────────────────────────
    results        = {}   # model_name -> val_f1
    test_preds_all = {}   # model_name -> (preds, labels)
    val_preds_all  = {}

    # ── Loop over models ──────────────────────────────────────────────────────
    for model_name, model_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}  ({model_path})")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model     = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=3, ignore_mismatched_sizes=True
        ).to(DEVICE)

        train_ds = NewsDataset(X_train, y_train, tokenizer, MAX_LEN)
        val_ds   = NewsDataset(X_val,   y_val,   tokenizer, MAX_LEN)
        test_ds  = NewsDataset(X_test,  y_test,  tokenizer, MAX_LEN)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        best_val_f1    = 0
        best_val_preds = None
        best_state     = None

        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, train_loader, optimizer, scheduler)
            val_preds, val_true = evaluate(model, val_loader)
            val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
            print(f"  Epoch {epoch}/{EPOCHS}  loss={loss:.4f}  val_F1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1    = val_f1
                best_val_preds = val_preds
                best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Restore best checkpoint and evaluate on test
        model.load_state_dict(best_state)
        model.to(DEVICE)
        test_preds, test_true = evaluate(model, test_loader)

        results[model_name]        = best_val_f1
        val_preds_all[model_name]  = (best_val_preds, np.array(y_val))
        test_preds_all[model_name] = (test_preds, np.array(y_test))

        print(f"\n  Best val F1-macro : {best_val_f1:.4f}")
        print(f"  Test classification report:")
        print(classification_report(
            test_true, test_preds,
            target_names=CLASS_NAMES, zero_division=0,
        ))

        # Free GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Select best model ─────────────────────────────────────────────────────
    best_model = max(results, key=results.get)
    print(f"\n{'='*60}")
    print(f"Best model by validation F1-macro: {best_model}  (F1={results[best_model]:.4f})")
    print(f"{'='*60}")

    # ── Plot confusion matrices ───────────────────────────────────────────────
    n_models = len(MODELS)
    fig, axes = plt.subplots(
        2, n_models,
        figsize=(5 * n_models, 10),
    )
    fig.suptitle(
        "Confusion matrices — Validation (top) and Test (bottom)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for col, model_name in enumerate(MODELS):
        # Validation
        val_p, val_t = val_preds_all[model_name]
        cm_val = confusion_matrix(val_t, val_p)
        plot_confusion_matrix(cm_val, model_name, "Validation", axes[0, col])

        # Test
        tst_p, tst_t = test_preds_all[model_name]
        cm_tst = confusion_matrix(tst_t, tst_p)
        plot_confusion_matrix(cm_tst, model_name, "Test", axes[1, col])

        # Highlight best model
        for row in range(2):
            if model_name == best_model:
                for spine in axes[row, col].spines.values():
                    spine.set_edgecolor("gold")
                    spine.set_linewidth(3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nConfusion matrices saved to '{plot_path}'")
    plt.show()

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = pd.DataFrame([
        {
            "Model"   : name,
            "Val F1"  : round(results[name], 4),
            "Best"    : "✓" if name == best_model else "",
        }
        for name in MODELS
    ])
    summary_path = os.path.join(OUTPUT_DIR, "model_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Model summary saved to '{summary_path}'")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()