"""Read the grids with every model, then solve the argmax readings."""
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from digits import DATA_DIR, MODELS, SEEDS, load_model, predict_logits
from generate import WORKERS, load_puzzles
from grids import cut_cells, draw_grid, grid_cells, puzzle_labels
from plots import INK, MODEL_COLORS, MUTED, save, style
from sudoku_sat import count_solutions, solve

RESULTS = Path("results")
OUTCOMES = ["perfect", "no solution", "several", "unique, right", "unique, wrong"]


def grid_logits(split: str) -> dict[tuple[str, int], torch.Tensor]:
    """Return {(model, seed): logits (n x 81 x 10)} for the grids of a split, cached in data/."""
    path = DATA_DIR / f"logits_{split}.pt"
    if path.exists():
        return torch.load(path)
    puzzles = load_puzzles(split)
    cells, _ = grid_cells(puzzles, split)
    cut = torch.stack([cut_cells(draw_grid(c)) for c in cells])  # what a reader of the images sees
    assert torch.equal(cut, cells)
    flat = cut.reshape(-1, 1, 28, 28)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits = {(name, seed): predict_logits(load_model(name, seed, device), flat).reshape(len(puzzles), 81, 10)
              for name in MODELS for seed in SEEDS}
    torch.save(logits, path)
    return logits


def to_grid(labels: torch.Tensor) -> list[list[int]]:
    """Turn 81 cell labels into a 9 x 9 grid of ints."""
    return [[int(v) for v in labels[9 * r:9 * r + 9]] for r in range(9)]


def classify(reading: list[list[int]], puzzle: list[list[int]], solution: list[list[int]]) -> str:
    """Return the outcome of solving a reading: one of OUTCOMES."""
    if reading == puzzle:
        return "perfect"
    found = solve(reading)
    if found is None:
        return "no solution"
    if count_solutions(reading) >= 2:
        return "several"
    return "unique, right" if found == solution else "unique, wrong"


def main() -> None:
    """Read and solve the grids of a split ('val' or 'test', from the command line) with all 9 models."""
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    suffix = "" if split == "test" else f"_{split}"
    out = RESULTS / f"reading_outcomes{suffix}.csv"
    if split == "test" and out.exists():
        print(f"{out} exists: the test grids are solved only once.")
        return
    puzzles = load_puzzles(split)
    labels = puzzle_labels(puzzles)
    digit = labels != 0
    logits = grid_logits(split)
    rows, per_grid = [], []
    with Pool(WORKERS) as pool:
        for (name, seed), z in logits.items():
            pred = z.argmax(2)
            correct = pred == labels
            # Value under independence: product over cells of the accuracy of the cell's true class.
            class_acc = torch.tensor([correct[labels == k].float().mean().item() for k in range(10)])
            predicted = class_acc[labels].prod(1).mean().item()
            jobs = [(to_grid(pred[i]), p["puzzle"], p["solution"]) for i, p in enumerate(puzzles)]
            outcomes = pool.starmap(classify, jobs)
            errors = (~correct).sum(1)
            per_grid += [dict(model=name, seed=seed, grid=i, errors=errors[i].item(), outcome=o)
                         for i, o in enumerate(outcomes)]
            row = dict(model=name, seed=seed, acc_digits=correct[digit].float().mean().item(),
                       acc_blanks=correct[~digit].float().mean().item(), predicted_perfect=predicted)
            row.update({o: outcomes.count(o) / len(puzzles) for o in OUTCOMES})
            rows.append(row)
            print(f"{name:6s} seed {seed}: digits {100 * row['acc_digits']:.2f}%, perfect {100 * row['perfect']:.1f}% "
                  f"(independence predicts {100 * predicted:.1f}%), unique wrong {outcomes.count('unique, wrong')}")
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)
    grids = pd.DataFrame(per_grid)

    print(f"\nMean over seeds (% of the {len(puzzles)} {split} grids):")
    print((100 * table.groupby("model", sort=False)[["acc_digits", "predicted_perfect"] + OUTCOMES].mean()).round(2)
          .to_string())
    print("\nOutcome of misread grids by number of misread cells (all models and seeds, % of row):")
    misread = grids[grids.errors > 0].assign(errors=grids.errors.clip(upper=4))
    by_errors = pd.crosstab(misread.errors, misread.outcome)
    by_errors = by_errors.reindex(columns=[o for o in OUTCOMES[1:] if o in by_errors.columns])
    total = by_errors.sum(axis=1)
    print(pd.concat([total.rename("grids"), (100 * by_errors.div(total, axis=0)).round(1)],
                    axis=1).rename(index={4: "4+"}).to_string())

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name in MODELS:
        t = table[table.model == name]
        ax.scatter(100 * t.predicted_perfect, 100 * t.perfect, s=60, color=MODEL_COLORS[name], label=name,
                   edgecolors="white", linewidths=1.5, zorder=3)
    ax.plot([0, 100], [0, 100], color=MUTED, linewidth=1, label="observed = predicted")
    style(ax)
    ax.set_xlabel("predicted by independence (%)", color=INK)
    ax.set_ylabel("grids read perfectly (%)", color=INK)
    ax.set_title(f"Grid accuracy: observed against independence\n({len(puzzles)} {split} grids, one dot per seed)",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, labelcolor=INK)
    save(fig, RESULTS / f"reading_independence{suffix}.png")


if __name__ == "__main__":
    main()
