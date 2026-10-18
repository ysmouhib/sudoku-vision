"""Draw sudoku images from cell images, and cut them back into cells."""
import torch

from digits import digit_pool, make_blanks

CELL = 28
LINE = 2
SIZE = 9 * CELL + 10 * LINE  # 272
LINE_VALUE = 1.0  # white lines, like the white strokes of MNIST digits on black
GRID_SEEDS = {"val": 10, "test": 20}  # choice of digit images
BLANK_SEED_BASE = {"val": 10_000, "test": 20_000}  # blank cells of grid i use seed base + i


def cell_origin(i: int) -> int:
    """Return the pixel offset of the i-th row (or column) of cells."""
    return LINE + i * (CELL + LINE)


def draw_grid(cells: torch.Tensor) -> torch.Tensor:
    """Paste 81 cells (81 x 1 x 28 x 28, row by row) into a 272 x 272 image with 2-pixel lines."""
    image = torch.full((SIZE, SIZE), LINE_VALUE)
    for k in range(81):
        y, x = cell_origin(k // 9), cell_origin(k % 9)
        image[y:y + CELL, x:x + CELL] = cells[k, 0]
    return image


def cut_cells(image: torch.Tensor) -> torch.Tensor:
    """Cut a 272 x 272 grid image back into its 81 cells (81 x 1 x 28 x 28)."""
    cells = [image[cell_origin(k // 9):cell_origin(k // 9) + CELL, cell_origin(k % 9):cell_origin(k % 9) + CELL]
             for k in range(81)]
    return torch.stack(cells).unsqueeze(1)


def grid_cells(puzzles: list[dict], split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the cell images of each puzzle from the split's digits; return cells (n x 81 x 1 x 28 x 28) and sources.

    sources[i, k] is the index of the digit image in digit_pool(split), or -1 for a blank cell.
    """
    images, labels = digit_pool(split)
    by_digit = {d: (labels == d).nonzero().flatten() for d in range(1, 10)}
    g = torch.Generator().manual_seed(GRID_SEEDS[split])
    cells = torch.empty(len(puzzles), 81, 1, CELL, CELL)
    sources = torch.full((len(puzzles), 81), -1, dtype=torch.long)
    for i, p in enumerate(puzzles):
        cells[i] = make_blanks(81, BLANK_SEED_BASE[split] + i)
        for k, d in enumerate(v for row in p["puzzle"] for v in row):
            if d:
                pool = by_digit[d]
                sources[i, k] = pool[torch.randint(len(pool), (1,), generator=g)].item()
                cells[i, k] = images[sources[i, k]]
    return cells, sources


def puzzle_labels(puzzles: list[dict]) -> torch.Tensor:
    """Return the true cell labels (n x 81): 0 for blank, else the clue."""
    return torch.tensor([[v for row in p["puzzle"] for v in row] for p in puzzles])


def main() -> None:
    """Save three validation grids as images, with their clue counts and digit reuse."""
    import matplotlib.pyplot as plt

    from generate import load_puzzles
    from plots import INK, save

    for split in ("val", "test"):
        puzzles = load_puzzles(split)
        _, sources = grid_cells(puzzles, split)
        used = sources[sources >= 0]
        counts = torch.bincount(used, minlength=len(digit_pool(split)[1]))
        print(f"{split}: {len(used)} digit cells; each digit image used {counts.float().mean():.2f} times on average, "
              f"{(counts == 0).float().mean():.1%} never used")
    puzzles = load_puzzles("val")[:3]
    cells, _ = grid_cells(puzzles, "val")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, p, c in zip(axes, puzzles, cells):
        ax.imshow(draw_grid(c), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{sum(v != 0 for row in p['puzzle'] for v in row)} clues", color=INK)
        ax.axis("off")
    save(fig, "results/grids_examples.png")


if __name__ == "__main__":
    main()
