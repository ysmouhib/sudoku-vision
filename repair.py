"""Repair misreadings with MaxSAT and a uniqueness loop."""
import sys
import threading
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from pysat.solvers import Solver

from digits import DATA_DIR, MODELS
from generate import WORKERS, load_puzzles
from sudoku_sat import clue_clauses, exactly_one, model_to_grid, rule_clauses, solve, var

N = 9
SCALE = 1000  # weights are -SCALE * log p, rounded: RC2 wants integers
MAX_TRIES = 20  # uniqueness loop: MaxSAT calls per grid at most
BUDGET = 10.0  # seconds per grid; a grid that runs out is a timeout: not solved
ANSWERS = ["right", "wrong", "timeout", "flagged"]
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
    with RC2(repair_formula(log_probs)) as rc2:
        return uniqueness_loop(rc2, max_tries, budget)[0]


def interrupt_from(rc2: RC2, deadline: float, done: threading.Event) -> None:
    """Interrupt RC2 at the deadline, then again every 0.05 s until done is set.

    One interrupt is not enough: an interrupt that arrives while RC2 is between two of its SAT calls is lost, and
    RC2 then runs without limit (it once ran for minutes and filled the memory).
    """
    if done.wait(max(0.0, deadline - time.perf_counter())):
        return
    while True:
        rc2.interrupt()
        if done.wait(0.05):
            return


def uniqueness_loop(rc2: RC2, max_tries: int, budget: float) -> tuple[list[list[list[int]]], str]:
    """Ask RC2 for readings in cost order, cutting those with several solutions, until one has exactly one.

    Return the readings tried and why the loop stopped: 'unique', 'max_tries', 'interrupted' (time budget) or
    'unsat' (no reading left). Afterwards rc2.cost is the cost of the last reading, or a lower bound on the cost of
    any reading not yet tried if the loop was interrupted.
    """
    readings = []
    deadline = time.perf_counter() + budget
    for _ in range(max_tries):
        if time.perf_counter() >= deadline:
            return readings, "interrupted"
        done = threading.Event()
        watcher = threading.Thread(target=interrupt_from, args=(rc2, deadline, done), daemon=True)
        watcher.start()
        model = rc2.compute(expect_interrupt=True)  # None if interrupted or unsatisfiable
        done.set()
        watcher.join()
        rc2.clear_interrupt()  # a late interrupt must not cut the next call short
        if model is None:
            return readings, "interrupted" if time.perf_counter() >= deadline else "unsat"
        readings.append(reading_from_model(model))
        solutions = two_solutions(readings[-1])
        if len(solutions) == 1:
            return readings, "unique"
        rc2.add_clause(pair_cut(*solutions))
    return readings, "max_tries"


def reading_cost(reading: list[list[int]], log_probs: list[list[float]]) -> int:
    """Return the total weight of a reading: the MaxSAT cost of choosing it."""
    return sum(weight(log_probs[9 * r + c][reading[r][c]]) for r in range(N) for c in range(N))


def repair_grid(log_probs: list[list[float]], puzzle: list[list[int]], solution: list[list[int]]) -> dict:
    """Repair one grid; return the correctness of each reading, the final answer and the time.

    The answer is 'right' or 'wrong' when a reading with a unique solution was found; 'timeout' when the time
    budget ran out first; 'flagged' when MAX_TRIES readings all had several solutions. Timeouts and flagged grids
    count as not solved.
    """
    start = time.perf_counter()
    with RC2(repair_formula(log_probs)) as rc2:
        readings, status = uniqueness_loop(rc2, MAX_TRIES, BUDGET)
    seconds = time.perf_counter() - start
    argmax = [[max(range(10), key=lambda k: log_probs[9 * r + c][k]) for c in range(N)] for r in range(N)]
    final = readings[-1] if readings else None
    if status == "unique":
        answer = "right" if solve(final) == solution else "wrong"
    else:
        answer = "timeout" if status == "interrupted" else "flagged"
    return dict(argmax_exact=argmax == puzzle, maxsat_exact=bool(readings) and readings[0] == puzzle,
                loop_exact=final == puzzle, answer=answer, tries=len(readings), seconds=seconds,
                final="".join(str(v) for row in final for v in row) if final else "")


def mark_timeouts(table: pd.DataFrame) -> pd.DataFrame:
    """Relabel as 'timeout' the 'flagged' grids whose loop stopped before MAX_TRIES readings.

    Needed for tables written before repair_grid told timeouts apart: the loop stops early only when it finds a
    unique reading or the budget runs out, so a flagged grid with fewer than MAX_TRIES readings is a timeout.
    Idempotent on newer tables.
    """
    early = (table.answer == "flagged") & (table.tries < MAX_TRIES)
    return table.assign(answer=table.answer.mask(early, "timeout"))


def run(logits: dict, puzzles: list[dict], label: str) -> pd.DataFrame:
    """Repair every grid with every model; return one row per (model, seed, grid)."""
    rows = []
    with Pool(WORKERS) as pool:
        for (name, seed), z in logits.items():
            log_probs = torch.log_softmax(z.double(), dim=2).tolist()
            jobs = [(log_probs[i], p["puzzle"], p["solution"]) for i, p in enumerate(puzzles)]
            start = time.perf_counter()
            for i, result in enumerate(pool.starmap(repair_grid, jobs, chunksize=8)):
                rows.append(dict(model=name, seed=seed, grid=i, probs=label, **result))
            print(f"{label} {name} seed {seed}: {time.perf_counter() - start:.0f} s", flush=True)
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Return the per-(model, seed) rates of exact readings and final answers, in % of grids."""
    table = mark_timeouts(table)
    g = table.groupby(["probs", "model", "seed"], sort=False)
    s = 100 * g[["argmax_exact", "maxsat_exact", "loop_exact"]].mean()
    for answer in ANSWERS:  # every grid is in exactly one of these: they add up to 100%
        s[f"answer_{answer}"] = 100 * g.answer.apply(lambda a: (a == answer).mean())
    s["grids"] = g.size()
    s["mean_seconds"] = g.seconds.mean()
    s["max_seconds"] = g.seconds.max()
    s["mean_tries"] = g.tries.mean()
    return s.reset_index()


def load_or_run(split: str, label: str, logits: dict) -> pd.DataFrame:
    """Return the per-grid repair table of a split and probability version, running it only the first time.

    The cache in data/ guarantees that each configuration runs on the test grids exactly once.
    """
    path = DATA_DIR / f"repair_{split}_{label}.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"final": str}, keep_default_na=False)
    table = run(logits, load_puzzles(split), label)
    table.to_csv(path, index=False)
    return table


def main() -> None:
    """Repair the grids of a split ('val' or 'test', from the command line) read by every model; write the results."""
    from read_grids import grid_logits

    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    summary = summarize(load_or_run(split, "raw", grid_logits(split)))
    summary.to_csv(RESULTS / ("repair_summary.csv" if split == "test" else f"repair_summary_{split}.csv"), index=False)
    cols = ["argmax_exact", "maxsat_exact", "loop_exact"] + [f"answer_{a}" for a in ANSWERS] + [
        "mean_seconds", "max_seconds", "mean_tries"]
    print(summary.round(2).to_string(index=False))
    print(f"\nMean over seeds ({split}, % of grids):")
    print(summary.groupby("model", sort=False)[cols].mean().reindex(MODELS).round(2).to_string())


if __name__ == "__main__":
    main()
