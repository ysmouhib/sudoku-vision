"""The whole pipeline as one function: sudoku image -> cells -> CNN -> MaxSAT repair -> solution."""
import time

import torch
from pysat.examples.rc2 import RC2

from digits import load_model, predict_logits
from generate import load_puzzles
from grids import SIZE, cut_cells, draw_grid, grid_cells
from read_grids import to_grid
from repair import BUDGET, MAX_TRIES, repair_formula, uniqueness_loop
from sudoku_sat import format_grid, solve

STATUS = {"unique": "solved", "interrupted": "timeout", "max_tries": "flagged", "unsat": "flagged"}


def read_and_solve(image: torch.Tensor, model: torch.nn.Module | None = None, budget: float = BUDGET) -> dict:
    """Read a 272 x 272 sudoku image (pixels in [0, 1], white ink on black) and solve it.

    Returns the argmax reading, the repaired reading, the solution, the status and the time in seconds. The status
    is 'solved' when a reading with exactly one solution was found, 'timeout' when the time budget ran out first,
    'flagged' when MAX_TRIES readings all had several solutions; the repaired reading and the solution are None
    unless the grid is solved. The model defaults to the CNN of seed 0.
    """
    start = time.perf_counter()
    if tuple(image.shape) != (SIZE, SIZE):
        raise ValueError(f"expected a {SIZE} x {SIZE} image, got {tuple(image.shape)}")
    model = model if model is not None else load_model("cnn", 0)
    log_probs = torch.log_softmax(predict_logits(model, cut_cells(image.float())).double(), dim=1)
    with RC2(repair_formula(log_probs.tolist())) as rc2:
        readings, status = uniqueness_loop(rc2, MAX_TRIES, budget)
    final = readings[-1] if status == "unique" else None
    return dict(reading=to_grid(log_probs.argmax(1)), final=final, solution=solve(final) if final else None,
                status=STATUS[status], seconds=time.perf_counter() - start)


def main() -> None:
    """Solve the first 20 test grid images and compare with the true solutions."""
    puzzles = load_puzzles("test")[:20]
    cells, _ = grid_cells(puzzles, "test")
    model = load_model("cnn", 0)
    right = 0
    for i, p in enumerate(puzzles):
        out = read_and_solve(draw_grid(cells[i]), model)
        right += out["solution"] == p["solution"]
        print(f"grid {i:2d}: {out['status']:7s} {'right' if out['solution'] == p['solution'] else 'not right':9s} "
              f"{out['seconds']:.2f} s")
        if i == 0 and out["solution"]:
            print(format_grid(out["solution"]))
    print(f"{right} of {len(puzzles)} right")


if __name__ == "__main__":
    main()
