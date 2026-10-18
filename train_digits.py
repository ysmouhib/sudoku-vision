"""Train the three digit classifiers (three seeds each) and save the models, curves and diagnostics."""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from digits import MODEL_DIR, MODELS, SEEDS, cell_dataset, digit_pool, make_model, model_path, predict_logits
from plots import INK, MODEL_COLORS, MUTED, save, style

EPOCHS = 5
BATCH = 128
LR = 1e-3
RESULTS = Path("results")


def evaluate(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """Return the mean cross-entropy and the accuracy of a model on a dataset."""
    logits = predict_logits(model, images)
    return F.cross_entropy(logits, labels).item(), (logits.argmax(1) == labels).float().mean().item()


def train(name: str, seed: int, train_set: tuple, val_set: tuple, device: str) -> tuple[nn.Module, list[dict]]:
    """Train one model with Adam on mini-batches; return it and its history (epoch 0 = before training)."""
    torch.manual_seed(seed)
    model = make_model(name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    X, y = train_set[0].to(device), train_set[1].to(device)
    history = []
    for epoch in range(EPOCHS + 1):
        if epoch > 0:
            model.train()
            perm = torch.randperm(len(y)).to(device)
            for i in range(0, len(y), BATCH):
                idx = perm[i:i + BATCH]
                loss = F.cross_entropy(model(X[idx]), y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        train_loss, train_acc = evaluate(model, *train_set)
        val_loss, val_acc = evaluate(model, *val_set)
        history.append(dict(model=name, seed=seed, epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc))
    return model, history


def plot_samples(images: torch.Tensor, labels: torch.Tensor, path: Path) -> None:
    """Save 20 random cells with their labels (0 = blank)."""
    idx = torch.randperm(len(labels), generator=torch.Generator().manual_seed(0))[:20]
    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for ax, i in zip(axes.flat, idx):
        ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title("blank" if labels[i] == 0 else str(labels[i].item()), color=INK, fontsize=10)
        ax.axis("off")
    save(fig, path)


def plot_curves(history: pd.DataFrame, path: Path) -> None:
    """Save training (dashed) and validation (solid) loss and error rate per epoch, seed 0."""
    fig, (ax_loss, ax_err) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in MODELS:
        h = history[(history.model == name) & (history.seed == 0)]
        color = MODEL_COLORS[name]
        ax_loss.plot(h.epoch, h.train_loss, "--", color=color, linewidth=2)
        ax_loss.plot(h.epoch, h.val_loss, "-", color=color, linewidth=2, label=name)
        ax_err.semilogy(h.epoch, 100 * (1 - h.train_acc), "--", color=color, linewidth=2)
        ax_err.semilogy(h.epoch, 100 * (1 - h.val_acc), "-", color=color, linewidth=2, label=name)
    for ax, ylabel in ((ax_loss, "cross-entropy"), (ax_err, "error rate (%)")):
        style(ax)
        ax.set_xlabel("epoch", color=INK)
        ax.set_ylabel(ylabel, color=INK)
        ax.legend(frameon=False, labelcolor=INK, title="solid: validation, dashed: training")
    ax_loss.set_ylim(0, 0.6)
    save(fig, path)


def plot_confusions(confusions: dict[str, np.ndarray], path: Path) -> None:
    """Save one confusion matrix per model (rows: true class, columns: predicted); color shows errors only."""
    fig, axes = plt.subplots(1, len(confusions), figsize=(6 * len(confusions), 5.5))
    classes = ["blank"] + [str(d) for d in range(1, 10)]
    for ax, (name, m) in zip(axes, confusions.items()):
        errors = m.astype(float).copy()
        np.fill_diagonal(errors, np.nan)
        ax.imshow(np.log1p(errors), cmap="Blues", vmin=0, vmax=np.log1p(np.nanmax(errors)))
        for i in range(10):
            for j in range(10):
                if i == j:
                    ax.text(j, i, m[i, j], ha="center", va="center", fontsize=7, color=MUTED)
                elif m[i, j]:
                    dark = errors[i, j] > np.nanmax(errors) / 3
                    ax.text(j, i, m[i, j], ha="center", va="center", fontsize=8, color="white" if dark else INK)
        ax.set_xticks(range(10), classes)
        ax.set_yticks(range(10), classes)
        ax.set_xlabel("predicted", color=INK)
        ax.set_ylabel("true", color=INK)
        ax.set_title(f"{name}, seed 0 (validation)", color=INK)
    save(fig, path)


def plot_worst(images: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor, path: Path) -> None:
    """Save the 30 validation mistakes with the lowest probability on the true class."""
    probs = logits.softmax(1)
    pred = probs.argmax(1)
    wrong = (pred != labels).nonzero().flatten()
    order = wrong[probs[wrong, labels[wrong]].argsort()][:30]
    fig, axes = plt.subplots(3, 10, figsize=(13, 4.8))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, order):
        ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{labels[i].item()}→{pred[i].item()} ({probs[i, pred[i]]:.2f})", color=INK, fontsize=9)
    save(fig, path)


def main() -> None:
    """Train every model and seed, then write the results and figures."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    RESULTS.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    for split in ("train", "val", "test"):
        assert (digit_pool(split)[1] != 0).all(), "an MNIST 0 is left"
    train_set, val_set = cell_dataset("train"), cell_dataset("val")
    for split, (_, labels) in (("train", train_set), ("val", val_set)):
        print(f"{split}: {len(labels)} cells, class counts (blank, 1..9):", torch.bincount(labels).tolist())
    plot_samples(*train_set, RESULTS / "digits_samples.png")

    history, models = [], {}
    for name in MODELS:
        for seed in SEEDS:
            start = time.perf_counter()
            model, h = train(name, seed, train_set, val_set, device)
            torch.save(model.state_dict(), model_path(name, seed))
            models[name, seed] = model
            history += h
            print(f"{name:6s} seed {seed}: val accuracy {h[-1]['val_acc']:.4f} ({time.perf_counter() - start:.0f} s)")
    history = pd.DataFrame(history)
    history.to_csv(RESULTS / "digits_history.csv", index=False)
    plot_curves(history, RESULTS / "digits_curves.png")

    rows, confusions = [], {}
    images, labels = val_set
    digit = labels != 0
    for (name, seed), model in models.items():
        logits = predict_logits(model, images)
        pred = logits.argmax(1)
        final = history[(history.model == name) & (history.seed == seed)].iloc[-1]
        rows.append(dict(model=name, seed=seed, train_acc=final.train_acc, val_acc=final.val_acc,
                         val_acc_digits=(pred[digit] == labels[digit]).float().mean().item(),
                         val_acc_blanks=(pred[~digit] == labels[~digit]).float().mean().item(),
                         val_loss=final.val_loss))
        if seed == 0:
            m = np.zeros((10, 10), dtype=int)
            np.add.at(m, (labels.numpy(), pred.numpy()), 1)
            confusions[name] = m
            if name == "cnn":
                plot_worst(images, labels, logits, RESULTS / "digits_worst_cnn.png")
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "digits_accuracy.csv", index=False)
    plot_confusions(confusions, RESULTS / "digits_confusion.png")

    summary = table.groupby("model", sort=False).agg(["mean", "min", "max"])
    print()
    for name in MODELS:
        s = summary.loc[name]
        print(f"{name:6s} val acc {100 * s.val_acc['mean']:.2f}% [{100 * s.val_acc['min']:.2f}, {100 * s.val_acc['max']:.2f}]"
              f"  digits {100 * s.val_acc_digits['mean']:.2f}%  blanks {100 * s.val_acc_blanks['mean']:.2f}%"
              f"  train acc {100 * s.train_acc['mean']:.2f}%")
    m = confusions["cnn"].copy()
    np.fill_diagonal(m, 0)
    top = np.dstack(np.unravel_index(np.argsort(-m, axis=None)[:5], m.shape))[0]
    print("cnn seed 0, most frequent confusions (true -> predicted: count):",
          ", ".join(f"{i}->{j}: {m[i, j]}" for i, j in top))


if __name__ == "__main__":
    main()
