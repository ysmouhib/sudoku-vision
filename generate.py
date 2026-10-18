"""Generate random sudoku puzzles with a unique solution."""
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

from pysat.solvers import Solver

from sudoku_sat import rule_clauses, solve, var

DATA_DIR = Path("data")
SPLITS = {"val": (200, 1), "test": (1000, 2)}  # split: (number of puzzles, seed)
WORKERS = 3  # worker processes per pool: with the main process, at most 4 Python processes


def shuffle_symmetries(grid: list[list[int]], rng: random.Random) -> list[list[int]]:
    """Apply a random symmetry: relabel the digits, permute bands, stacks, rows and columns inside them, maybe transpose."""
    relabel = [0] + rng.sample(range(1, 10), 9)
    rows = [3 * b + r for b in rng.sample(range(3), 3) for r in rng.sample(range(3), 3)]
    cols = [3 * b + c for b in rng.sample(range(3), 3) for c in rng.sample(range(3), 3)]
    out = [[relabel[grid[r][c]] for c in cols] for r in rows]
    return [list(row) for row in zip(*out)] if rng.random() < 0.5 else out


def random_full_grid(rng: random.Random) -> list[list[int]]:
    """Return a random complete grid: random diagonal boxes, completed by the SAT solver, then a random symmetry."""
    grid = [[0] * 9 for _ in range(9)]
    for b in range(3):  # the three diagonal boxes share no row, column or box, so any filling is allowed
        digits = rng.sample(range(1, 10), 9)
        for i in range(9):
            grid[3 * b + i // 3][3 * b + i % 3] = digits[i]
    return shuffle_symmetries(solve(grid), rng)


def remove_clues(solution: list[list[int]], rng: random.Random, stop_at: int = 0) -> list[list[int]]:
    """Remove clues in random order while the solution stays unique (down to stop_at clues); return the puzzle."""
    clues = {(r, c) for r in range(9) for c in range(9)}
    order = sorted(clues)
    rng.shuffle(order)
    with Solver(name="g4", bootstrap_with=rule_clauses(9)) as solver:
        for r, c in order:
            if len(clues) <= stop_at:
                break
            others = [var(i, j, solution[i][j], 9) for i, j in clues if (i, j) != (r, c)]
            # Any other solution must differ from `solution` at the removed cell (else it would solve the puzzle).
            if not solver.solve(assumptions=others + [-var(r, c, solution[r][c], 9)]):
                clues.remove((r, c))
    return [[solution[r][c] if (r, c) in clues else 0 for c in range(9)] for r in range(9)]


def make_puzzle(seed: int, stop_at: int = 0) -> dict:
    """Return one puzzle and its solution, fully determined by the seed."""
    rng = random.Random(seed)
    solution = random_full_grid(rng)
    return {"puzzle": remove_clues(solution, rng, stop_at), "solution": solution}


def make_puzzles(n: int, seed: int, stop_at: int = 0) -> list[dict]:
    """Return n puzzles, generated in parallel; puzzle i uses seed 100000 * seed + i."""
    with Pool(WORKERS) as pool:
        return pool.starmap(make_puzzle, [(100_000 * seed + i, stop_at) for i in range(n)])


def puzzle_path(split: str) -> Path:
    """Return the file holding the puzzles of a split."""
    return DATA_DIR / f"puzzles_{split}.json"


def load_puzzles(split: str) -> list[dict]:
    """Load the puzzles of a split ('val' or 'test'), generating and saving them the first time."""
    path = puzzle_path(split)
    if not path.exists():
        n, seed = SPLITS[split]
        DATA_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(make_puzzles(n, seed)))
    return json.loads(path.read_text())


def main() -> None:
    """Generate the validation and test puzzles and print their clue counts."""
    for split, (n, seed) in SPLITS.items():
        start = time.perf_counter()
        puzzles = make_puzzles(n, seed)
        elapsed = time.perf_counter() - start
        DATA_DIR.mkdir(exist_ok=True)
        puzzle_path(split).write_text(json.dumps(puzzles))
        clues = sorted(sum(v != 0 for row in p["puzzle"] for v in row) for p in puzzles)
        print(f"{split}: {n} puzzles in {elapsed:.1f} s; clues min {clues[0]}, median {clues[n // 2]}, max {clues[-1]}")


if __name__ == "__main__":
    main()
