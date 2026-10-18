"""Draw the demo figure: image, reading (misreadings in red), repaired reading, solution."""
import matplotlib.pyplot as plt
import torch

from generate import load_puzzles
from grids import draw_grid, grid_cells
from plots import BAD, GOOD, INK, MUTED, save
from read_grids import grid_logits, to_grid
from repair import repair
from sudoku_sat import solve


def draw_digits(ax: plt.Axes, grid: list[list[int]], colors: dict[tuple[int, int], str], title: str) -> None:
    """Draw a sudoku grid as text; cells listed in colors get that color (bold if red or green)."""
    for i in range(10):
        width = 2 if i % 3 == 0 else 0.6
        ax.plot([i, i], [0, 9], color=INK, linewidth=width)
        ax.plot([0, 9], [i, i], color=INK, linewidth=width)
    for r in range(9):
        for c in range(9):
            if grid[r][c]:
                color = colors.get((r, c), INK)
                ax.text(c + 0.5, 8.5 - r, str(grid[r][c]), ha="center", va="center", fontsize=13, color=color,
                        fontweight="bold" if color in (BAD, GOOD) else "normal")
    ax.set_xlim(-0.1, 9.1)
    ax.set_ylim(-0.1, 9.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, color=INK, fontsize=11)


def main() -> None:
    """Find a test grid that the CNN misreads and the repair fixes, and draw the four stages."""
    puzzles = load_puzzles("test")
    logits = grid_logits("test")["cnn", 0]
    for i, p in enumerate(puzzles):
        reading = to_grid(logits[i].argmax(1))
        if reading != p["puzzle"] and solve(reading) is None:
            repaired = repair(torch.log_softmax(logits[i].double(), 1).tolist())[-1]
            if repaired == p["puzzle"]:
                break
    cells, _ = grid_cells(puzzles[:i + 1], "test")  # one random stream over the puzzles in order: replay it up to i
    probs = logits[i].softmax(1)
    misread = {(r, c) for r in range(9) for c in range(9) if reading[r][c] != p["puzzle"][r][c]}
    for r, c in sorted(misread):
        k = 9 * r + c
        print(f"grid {i}, cell ({r}, {c}): true {p['puzzle'][r][c]}, read {reading[r][c]} "
              f"(p = {probs[k, reading[r][c]]:.3f}; true digit p = {probs[k, p['puzzle'][r][c]]:.3f})")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    axes[0].imshow(draw_grid(cells[i]), cmap="gray", vmin=0, vmax=1)
    axes[0].axis("off")
    axes[0].set_title(f"1. Image (test grid {i})", color=INK, fontsize=11)
    draw_digits(axes[1], reading, {cell: BAD for cell in misread}, "2. CNN reading: no solution (red: misread)")
    draw_digits(axes[2], repaired, {cell: GOOD for cell in misread}, "3. MaxSAT repair (green: corrected)")
    givens = {(r, c) for r in range(9) for c in range(9) if p["puzzle"][r][c]}
    draw_digits(axes[3], p["solution"], {cell: MUTED for r in range(9) for c in range(9) if (cell := (r, c)) not in givens},
                "4. Solution (grey: filled in by SAT)")
    save(fig, "results/demo.png")


if __name__ == "__main__":
    main()
