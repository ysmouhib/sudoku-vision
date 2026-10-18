"""Fit calibrations on validation cells, choose one on the validation grids, and run it once on the test grids."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from digits import DATA_DIR, MODELS, SEEDS, cell_dataset, load_model, predict_logits
from generate import load_puzzles
from grids import puzzle_labels
from plots import INK, MUTED, save, style
from read_grids import grid_logits
from repair import ANSWERS, load_or_run, summarize

RESULTS = Path("results")
T_GRID = torch.arange(50, 501) / 100  # 0.50, 0.51, ..., 5.00
N_BINS = 15
METHODS = ["raw", "temperature", "vector"]
METHOD_COLORS = {"raw": "#2a78d6", "temperature": "#eb6834", "vector": "#1baf7a"}


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the T on the grid 0.5..5 (step 0.01) minimizing the cross-entropy of softmax(z / T)."""
    losses = torch.stack([F.cross_entropy(logits / t, labels) for t in T_GRID])
    return T_GRID[losses.argmin()].item()


def fit_vector_scaling(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-class scales a and biases b minimizing the cross-entropy of softmax(a * z + b), by L-BFGS."""
    a = torch.ones(10, requires_grad=True)
    b = torch.zeros(10, requires_grad=True)
    optimizer = torch.optim.LBFGS([a, b], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits * a + b, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return a.detach(), b.detach()


def apply(z: torch.Tensor, fit: tuple, method: str) -> torch.Tensor:
    """Return the logits of a calibration method: raw z, z / T, or a * z + b."""
    t, a, b = fit
    return {"raw": z, "temperature": z / t, "vector": z * a + b}[method]


def reliability(probs: torch.Tensor, labels: torch.Tensor) -> tuple[float, list[tuple[float, float, int]]]:
    """Return the expected calibration error (15 equal bins of confidence) and the (confidence, accuracy, count) bins."""
    conf, pred = probs.max(1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, N_BINS + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = (conf[m].mean() - correct[m].mean()).abs().item()
            ece += m.float().mean().item() * gap
            bins.append((conf[m].mean().item(), correct[m].mean().item(), int(m.sum())))
    return ece, bins


def quality(z: torch.Tensor, labels: torch.Tensor) -> dict:
    """Return the calibration error (%), the cross-entropy and the accuracy (%) of logits z."""
    return dict(ece=100 * reliability(z.softmax(1), labels)[0], nll=F.cross_entropy(z, labels).item(),
                acc=100 * (z.argmax(1) == labels).float().mean().item())


def plot_reliability(panels: dict[str, dict[str, tuple]], title: str, path: Path) -> None:
    """Save one reliability diagram per model: {model: {method: (logits, labels)}}."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    for ax, (name, versions) in zip(axes, panels.items()):
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1)
        for method, (z, labels) in versions.items():
            ece, bins = reliability(z.softmax(1), labels)
            ax.plot([c for c, _, _ in bins], [acc for _, acc, _ in bins], "o-", color=METHOD_COLORS[method],
                    linewidth=2, markersize=5, label=f"{method} (ECE {100 * ece:.2f}%)")
        style(ax)
        ax.set_xlabel("confidence (max probability)", color=INK)
        ax.set_ylabel("accuracy", color=INK)
        ax.set_title(f"{name}, seed 0: {title}", color=INK, fontsize=11)
        ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    save(fig, path)


def changed_finals(table: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """Return, per (model, seed), the number of grids whose final reading differs from the reference table."""
    key = ["model", "seed", "grid"]
    other = table.set_index(key).final
    return (other != reference.set_index(key).final.reindex(other.index)).groupby(["model", "seed"]).sum()


def main() -> None:
    """Fit on validation cells, choose on validation grids, run the chosen method once on the test grids."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_images, val_labels = cell_dataset("val")
    fits, rows, val_panels = {}, [], {}
    for name in MODELS:
        for seed in SEEDS:
            zv = predict_logits(load_model(name, seed, device), val_images)
            fits[name, seed] = (fit_temperature(zv, val_labels), *fit_vector_scaling(zv, val_labels))
            row = dict(model=name, seed=seed, T=fits[name, seed][0])
            for method in METHODS:
                z = apply(zv, fits[name, seed], method)
                row.update({f"val_{k}_{method}": v for k, v in quality(z, val_labels).items()})
                if seed == 0:
                    val_panels.setdefault(name, {})[method] = (z, val_labels)
            assert row["val_acc_temperature"] == row["val_acc_raw"]  # T never changes the argmax
            rows.append(row)
            print(f"{name:6s} seed {seed}: T = {row['T']:.2f}, validation ECE {row['val_ece_raw']:.2f}% (raw) "
                  f"{row['val_ece_temperature']:.2f}% (T) {row['val_ece_vector']:.2f}% (vector)", flush=True)
    calib = pd.DataFrame(rows)
    torch.save(fits, DATA_DIR / "calibration_fits.pt")  # (T, a, b) per model, reused by undetectable.py
    plot_reliability(val_panels, "validation cells (fit data)", RESULTS / "calibration_reliability_val.png")

    # Choose the method on the validation grids.
    val_logits = grid_logits("val")
    val_tables = {m: load_or_run("val", m, {k: apply(z, fits[k], m) for k, z in val_logits.items()}) for m in METHODS}
    val_summary = summarize(pd.concat(val_tables.values(), ignore_index=True))
    for m in METHODS[1:]:
        for (name, seed), n in changed_finals(val_tables[m], val_tables["raw"]).items():
            val_summary.loc[(val_summary.probs == m) & (val_summary.model == name) & (val_summary.seed == seed),
                            "changed"] = n
    val_summary.to_csv(RESULTS / "calibration_repair_val.csv", index=False)
    score = val_summary.groupby("probs")[["answer_right", "answer_wrong"]].mean()
    score["val_nll"] = [calib[f"val_nll_{m}"].mean() for m in score.index]
    chosen = min(METHODS, key=lambda m: (-score.loc[m, "answer_right"], score.loc[m, "answer_wrong"],
                                         score.loc[m, "val_nll"]))
    print("\nValidation grids, mean over the 9 models (choice: most right answers, then fewest wrong, then lowest "
          "validation cross-entropy):")
    print(score.round(3).to_string())
    print(f"Chosen: {chosen}")

    # Run the chosen configuration on the test grids, once (cached in data/ by load_or_run).
    test_logits = grid_logits("test")
    raw_test = load_or_run("test", "raw", test_logits)
    final_test = load_or_run("test", chosen, {k: apply(z, fits[k], chosen) for k, z in test_logits.items()})
    test_summary = summarize(pd.concat([raw_test, final_test] if chosen != "raw" else [raw_test], ignore_index=True))
    if chosen != "raw":
        for (name, seed), n in changed_finals(final_test, raw_test).items():
            test_summary.loc[(test_summary.probs == chosen) & (test_summary.model == name)
                             & (test_summary.seed == seed), "changed"] = n
    test_summary.to_csv(RESULTS / "calibration_repair_test.csv", index=False)

    labels = puzzle_labels(load_puzzles("test")).reshape(-1)
    digit = labels != 0
    test_panels = {}
    for row in rows:
        z = test_logits[row["model"], row["seed"]].reshape(-1, 10)[digit]
        for method in dict.fromkeys(["raw", chosen]):
            zz = apply(z, fits[row["model"], row["seed"]], method)
            row.update({f"test_{k}_{method}": v for k, v in quality(zz, labels[digit]).items()})
            if row["seed"] == 0:
                test_panels.setdefault(row["model"], {})[method] = (zz, labels[digit])
    calib = pd.DataFrame(rows).assign(chosen=chosen)
    calib.to_csv(RESULTS / "calibration_summary.csv", index=False)
    plot_reliability(test_panels, "test digit cells", RESULTS / "calibration_reliability_test.png")

    cols = ["loop_exact"] + [f"answer_{a}" for a in ANSWERS] + ["changed", "mean_seconds"]
    print("\nCalibration (mean over seeds):")
    print(calib.groupby("model", sort=False).mean(numeric_only=True).drop(columns="seed").round(3).to_string())
    print("\nRepair on the validation grids (mean over seeds):")
    print(val_summary.groupby(["probs", "model"], sort=False)[cols].mean().round(2).to_string())
    print(f"\nRepair on the test grids (mean over seeds), raw and the chosen method ({chosen}):")
    print(test_summary.groupby(["probs", "model"], sort=False)[[c for c in cols if c in test_summary]].mean()
          .round(2).to_string())


if __name__ == "__main__":
    main()
