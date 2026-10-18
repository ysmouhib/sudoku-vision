"""Repair misreadings with MaxSAT and a uniqueness loop."""
import threading
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from pysat.solvers import Solver

from digits import MODELS
from generate import load_puzzles
from sudoku_sat import clue_clauses, count_solutions, exactly_one, model_to_grid, rule_clauses, solve, var

N = 9
SCALE = 1000  # weights are -SCALE * log p, rounded: RC2 wants integers
MAX_TRIES = 20  # uniqueness loop: MaxSAT calls per grid at most
BUDGET = 10.0  # seconds per grid; a grid that runs out is flagged (no answer)
RESULTS = Path("results")


def y(r: int, c: int, k: int) -> int:
    """Return the variable 'cell (r, c) is read as k' (k = 0 for blank), numbered after the 729 x's."""
    return N**3 + (r * N + c) * 10 + k + 1


def weight(log_p: float) -> int:
    """Return the weight of the soft clause of a reading with log-probability log_p: max(1, round(-1000 log p))."""
    return max(1, round(-SCALE * log_p))


def repair_formula(log_probs: list[list[float]]) -> WCNF:
    """Return the MaxSAT formula of one grid; log_probs[9r + c][k] = log p(cell (r, c) is read as k).

    Soft weights are max(1, round(-1000 log p)) minus each cell's smallest weight. Every reading picks exactly one
    class per cell, so every reading's cost drops by the same constant (cost_offset): the order of the readings,
    and hence the optimum and the uniqueness loop, are unchanged. Each cell's cheapest class then needs no
    soft clause, which removes 81 trivial cores and makes RC2 about 100 times faster.
    """
    wcnf = WCNF()
    for clause in rule_clauses(N):  # hard: the sudoku rules on x
        wcnf.append(clause)
    for r in range(N):
        for c in range(N):
            for clause in exactly_one([y(r, c, k) for k in range(10)]):  # hard: one reading per cell
                wcnf.append(clause)
            for k in range(1, 10):  # hard: if the cell is read as k, the solution has k there
                wcnf.append([-y(r, c, k), var(r, c, k, N)])
            w = [weight(lp) for lp in log_probs[9 * r + c]]
            for k in range(10):  # soft: choosing reading k costs w(k) - min w
                if w[k] > min(w):
                    wcnf.append([-y(r, c, k)], weight=w[k] - min(w))
    return wcnf


def cost_offset(log_probs: list[list[float]]) -> int:
    """Return the constant removed from every reading's cost: the sum over cells of the smallest weight."""
    return sum(min(weight(lp) for lp in cell) for cell in log_probs)


def reading_from_model(model: list[int]) -> list[list[int]]:
    """Return the reading (0 = blank) chosen by a MaxSAT model."""
    true = {lit for lit in model if lit > 0}
    return [[next(k for k in range(10) if y(r, c, k) in true) for c in range(N)] for r in range(N)]


def two_solutions(grid: list[list[int]]) -> list[list[list[int]]]:
    """Return up to two solutions of a grid (0 = empty cell)."""
    found = []
    with Solver(name="g4", bootstrap_with=rule_clauses(N) + clue_clauses(grid)) as solver:
        while len(found) < 2 and solver.solve():
            found.append(model_to_grid(solver.get_model(), N))
            solver.add_clause([-var(r, c, found[-1][r][c], N) for r in range(N) for c in range(N)])
    return found


def pair_cut(a: list[list[int]], b: list[list[int]]) -> list[int]:
    """Return a clause excluding every reading that both solutions a and b satisfy.

    A reading has a and b as solutions iff each of its clues k at (r, c) has a[r][c] == b[r][c] == k. So the
    clause asks for at least one other clue. It only removes readings with two or more solutions.
    """
    return [y(r, c, k) for r in range(N) for c in range(N) for k in range(1, 10) if not a[r][c] == b[r][c] == k]


def repair(log_probs: list[list[float]], max_tries: int = MAX_TRIES, budget: float = BUDGET) -> list[list[list[int]]]:
    """Return the readings tried, best first: the first is the MaxSAT optimum (the most probable reading
    consistent with a valid grid); the last is the most probable reading with exactly one solution, unless
    max_tries or the time budget (seconds) ran out first.

    Uniqueness loop: when a reading has two solutions a and b, add pair_cut(a, b). This excludes the reading
    itself (as the plain blocking clause does) and every other reading that a and b both satisfy. Every excluded
    reading has two or more solutions, so the loop ends on the same reading as with the plain blocking clause,
    only in fewer calls.
    """
    readings = []
    deadline = time.perf_counter() + budget
    with RC2(repair_formula(log_probs)) as rc2:
        for _ in range(max_tries):
            timer = threading.Timer(max(0.0, deadline - time.perf_counter()), rc2.interrupt)
            timer.start()
            model = rc2.compute(expect_interrupt=True)  # None if interrupted
            timer.cancel()
            if model is None:
                break
            readings.append(reading_from_model(model))
            solutions = two_solutions(readings[-1])
            if len(solutions) == 1:
                break
            rc2.add_clause(pair_cut(*solutions))
    return readings


def reading_cost(reading: list[list[int]], log_probs: list[list[float]]) -> int:
    """Return the total weight of a reading: the MaxSAT cost of choosing it."""
    return sum(weight(log_probs[9 * r + c][reading[r][c]]) for r in range(N) for c in range(N))


def repair_grid(log_probs: list[list[float]], puzzle: list[list[int]], solution: list[list[int]]) -> dict:
    """Repair one grid and return what happened: correctness of each reading and of the final answer, and the time."""
    start = time.perf_counter()
    readings = repair(log_probs)
    seconds = time.perf_counter() - start
    argmax = [[max(range(10), key=lambda k: log_probs[9 * r + c][k]) for c in range(N)] for r in range(N)]
    final = readings[-1] if readings else None
    if final is None or count_solutions(final) != 1:
        answer = "flagged"  # no reading with a unique solution within the tries and the time budget
    else:
        answer = "right" if solve(final) == solution else "wrong"
    return dict(argmax_exact=argmax == puzzle, maxsat_exact=bool(readings) and readings[0] == puzzle,
                loop_exact=final == puzzle, answer=answer, tries=len(readings), seconds=seconds)


def run(logits: dict, puzzles: list[dict], label: str) -> pd.DataFrame:
    """Repair every grid with every model; return one row per (model, seed, grid)."""
    rows = []
    with Pool() as pool:
        for (name, seed), z in logits.items():
            log_probs = torch.log_softmax(z.double(), dim=2).tolist()
            jobs = [(log_probs[i], p["puzzle"], p["solution"]) for i, p in enumerate(puzzles)]
            for i, result in enumerate(pool.starmap(repair_grid, jobs, chunksize=8)):
                rows.append(dict(model=name, seed=seed, grid=i, probs=label, **result))
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Return the per-(model, seed) rates of exact readings and final answers, in % of grids."""
    g = table.groupby(["probs", "model", "seed"], sort=False)
    s = 100 * g[["argmax_exact", "maxsat_exact", "loop_exact"]].mean()
    for answer in ("right", "wrong", "flagged"):
        s[f"answer_{answer}"] = 100 * g.answer.apply(lambda a: (a == answer).mean())
    s["mean_seconds"] = g.seconds.mean()
    s["max_seconds"] = g.seconds.max()
    s["mean_tries"] = g.tries.mean()
    return s.reset_index()


def main() -> None:
    """Repair the test grids read by every model; write the results."""
    from read_grids import grid_logits

    table = run(grid_logits("test"), load_puzzles("test"), "raw")
    summary = summarize(table)
    summary.to_csv(RESULTS / "repair_summary.csv", index=False)
    cols = ["argmax_exact", "maxsat_exact", "loop_exact", "answer_right", "answer_wrong", "answer_flagged",
            "mean_seconds", "max_seconds", "mean_tries"]
    print(summary.round(2).to_string(index=False))
    print("\nMean over seeds:")
    print(summary.groupby("model", sort=False)[cols].mean().reindex(MODELS).round(2).to_string())


if __name__ == "__main__":
    main()
