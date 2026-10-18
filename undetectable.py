"""Count the clue substitutions that give a unique wrong solution, and flag doubtful answers by their margin."""
import math
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pysat.examples.rc2 import RC2
from pysat.solvers import Solver

from digits import DATA_DIR, MODELS
from generate import WORKERS, load_puzzles, make_puzzles
from plots import BAD, INK, MODEL_COLORS, MUTED, save, style
from repair import BUDGET, MAX_TRIES, N, repair_formula, uniqueness_loop, y
from sudoku_sat import model_to_grid, rule_clauses, solve, var

RESULTS = Path("results")
CENSUS_SETS = {"minimal": (1000, 3, 0), "30 clues": (1000, 4, 30)}  # name: (puzzles, seed, stop_at)
CAP = 1000  # solutions of P without c enumerated at most
R = 10  # an undetected wrong answer costs R times as much as declining to answer a right one
TAU_GRID = np.round(np.arange(0, 20.001, 0.05), 2)  # candidate thresholds on the margin (nats)


def peers(r: int, c: int) -> set[tuple[int, int]]:
    """Return the 20 cells sharing a row, column or box with (r, c)."""
    box = {(3 * (r // 3) + i, 3 * (c // 3) + j) for i in range(3) for j in range(3)}
    return ({(r, j) for j in range(N)} | {(i, c) for i in range(N)} | box) - {(r, c)}


def clue_fibers(puzzle: list[list[int]]) -> list[dict]:
    """For each clue c of a puzzle with a unique solution: the solutions of the puzzle without c (m, capped at CAP)
    and, for each digit d, the size of its fiber {solutions with d at c}, capped at 2.

    Fiber lemma: the solutions of the puzzle with c replaced by d are exactly that fiber. So the substitution
    has no solution, a unique (wrong) solution, or several, as the fiber has 0, 1 or 2+ elements.
    """
    clues = [(r, c) for r in range(N) for c in range(N) if puzzle[r][c]]
    rows, guard = [], N**3
    with Solver(name="g4", bootstrap_with=rule_clauses(N)) as solver:

        def enumerate_(assumptions: list[int], limit: int) -> list[list[list[int]]]:
            """Return up to limit solutions under the assumptions, blocking each with a clause guarded by a fresh literal."""
            nonlocal guard
            guard += 1
            found = []
            while len(found) < limit and solver.solve(assumptions=assumptions + [guard]):
                grid = model_to_grid([lit for lit in solver.get_model() if abs(lit) <= N**3], N)
                found.append(grid)
                solver.add_clause([-guard] + [-var(i, j, grid[i][j], N) for i in range(N) for j in range(N)])
            return found

        for r, c in clues:
            others = [var(i, j, puzzle[i][j], N) for i, j in clues if (i, j) != (r, c)]
            solutions = enumerate_(others, CAP)
            fiber = [0] * 10
            for s in solutions:
                fiber[s[r][c]] = min(2, fiber[s[r][c]] + 1)
            if len(solutions) == CAP:  # the enumeration was cut: recount the small fibers exactly (up to 2)
                for d in range(1, 10):
                    if fiber[d] <= 1:
                        fiber[d] = len(enumerate_(others + [var(r, c, d, N)], 2))
            rows.append(dict(r=r, c=c, digit=puzzle[r][c], m=len(solutions), capped=len(solutions) == CAP,
                             peer_clues=sum(puzzle[i][j] != 0 for i, j in peers(r, c)),
                             **{f"fiber{d}": fiber[d] for d in range(1, 10)}))
    return rows


def census_rows(i: int, p: dict, set_name: str) -> list[dict]:
    """Return the census rows of one puzzle: one row per (clue, substituted digit)."""
    rows = []
    n_clues = sum(v != 0 for row in p["puzzle"] for v in row)
    for f in clue_fibers(p["puzzle"]):
        for d in range(1, 10):
            if d != f["digit"]:
                outcome = ["no solution", "unique", "several"][f[f"fiber{d}"]]
                rows.append(dict(set=set_name, puzzle=i, clues=n_clues, r=f["r"], c=f["c"], true=f["digit"],
                                 read=d, outcome=outcome, m=f["m"], capped=f["capped"], peer_clues=f["peer_clues"]))
    return rows


def margin(log_probs: list[list[float]], solution: list[list[int]], budget: float = BUDGET) -> dict:
    """Repair a grid, then find the best competing answer: the most probable reading whose unique solution is not
    the answer's. Return the answer's status, the margin in nats (or a lower bound) and the simpler next-reading rule."""
    with RC2(repair_formula(log_probs)) as rc2:
        readings, status = uniqueness_loop(rc2, MAX_TRIES, budget)
        if status != "unique":  # not solved: counted, never dropped
            return dict(answer="timeout" if status == "interrupted" else "flagged")
        r1, cost1 = readings[-1], rc2.cost  # costs are in the shifted weights; differences are unaffected
        s1 = solve(r1)
        result = dict(answer="right" if s1 == solution else "wrong")

        # The next-reading rule: is the next most probable reading also uniquely solvable, with another solution?
        rc2.add_clause([-y(r, c, r1[r][c]) for r in range(N) for c in range(N)])
        nxt, nxt_status = uniqueness_loop(rc2, 1, budget)
        result["next_flag"] = nxt_status == "unique" and solve(nxt[-1]) != s1

        # The margin: forbid the answer's solution grid, and continue the uniqueness loop from there.
        rc2.add_clause([-var(r, c, s1[r][c], N) for r in range(N) for c in range(N)])
        rivals, rival_status = uniqueness_loop(rc2, MAX_TRIES, budget)
        if rival_status == "unsat":  # no other answer is consistent with any reading at all
            result["margin"] = math.inf
        else:
            result["margin"] = (rc2.cost - cost1) / 1000  # exact if a rival was found, else a lower bound
        result["margin_exact"] = rival_status in ("unique", "unsat")
        result["rival_is_truth"] = rival_status == "unique" and solve(rivals[-1]) == solution
    return result


def run_census() -> pd.DataFrame:
    """Return the census of every clue substitution in fresh puzzles (minimal and 30-clue), cached in data/."""
    path = DATA_DIR / "census.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for set_name, (n, seed, stop_at) in CENSUS_SETS.items():
        puzzles = make_puzzles(n, seed, stop_at)
        with Pool(WORKERS) as pool:
            for part in pool.starmap(census_rows, [(i, p, set_name) for i, p in enumerate(puzzles)], chunksize=4):
                rows += part
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False)
    return table


def chi2_uniform(counts: np.ndarray, totals: np.ndarray) -> tuple[float, int, float]:
    """Test whether the rate counts / totals is the same in every cell; return chi^2, degrees of freedom, p-value."""
    expected = totals * counts.sum() / totals.sum()
    stat = float(((counts - expected) ** 2 / expected).sum())
    dof = len(counts) - 1
    p = torch.special.gammaincc(torch.tensor(dof / 2, dtype=torch.float64), torch.tensor(stat / 2, dtype=torch.float64))
    return stat, dof, float(p)


def analyze_census(census: pd.DataFrame) -> None:
    """Print and save the census rates: overall, by digit pair, by m, by peer clues, by clue count."""
    census = census.assign(unique=census.outcome == "unique")
    overall = 100 * pd.crosstab(census.set, census.outcome, normalize="index")
    print("Census, % of clue substitutions:\n" + overall.round(2).to_string())
    per_puzzle = census.groupby(["set", "puzzle"]).unique.mean()
    for set_name, rates in per_puzzle.groupby(level=0):
        print(f"{set_name}: per-puzzle undetectable rate: median {100 * rates.median():.2f}%, "
              f"share of puzzles with none {100 * (rates == 0).mean():.1f}%, max {100 * rates.max():.1f}%")

    minimal = census[census.set == "minimal"]
    pairs = minimal.groupby(["true", "read"]).unique.agg(["sum", "count"])
    stat, dof, p = chi2_uniform(pairs["sum"].to_numpy(), pairs["count"].to_numpy())
    rate = pairs["sum"] / pairs["count"]
    print(f"Minimal puzzles, by digit pair (72 pairs): rate {100 * rate.min():.2f}%..{100 * rate.max():.2f}%; "
          f"chi^2 = {stat:.1f} on {dof} dof, p = {p:.3f}")

    buckets = pd.cut(census.m, [1, 2, 3, 5, 10, 30, 100, CAP - 1, CAP], labels=["2", "3", "4-5", "6-10", "11-30",
                                                                             "31-100", "101-999", "1000+"])
    by_m = census[census.m >= 2].groupby(["set", buckets[census.m >= 2]], observed=True).unique.agg(["mean", "count"])
    by_m["mean"] *= 100
    by_m["count"] //= 8  # clues, not substitutions
    print("Undetectable rate (%) by m = |Sol(P without c)| (count = clues):\n" + by_m.round(2).to_string())
    redundant = census[census.m == 1]
    print(f"Redundant clues (m = 1): {len(redundant) // 8}, all substitutions give no solution: "
          f"{(redundant.outcome == 'no solution').all()}")

    by_peers = minimal.groupby("peer_clues").unique.agg(["mean", "count"])
    by_clues = minimal.groupby("clues").unique.agg(["mean", "count"])
    for label, t in (("peer clues", by_peers), ("clues in the puzzle", by_clues)):
        t = t[t["count"] >= 800]
        print(f"Minimal puzzles, undetectable rate (%) by {label}: "
              + ", ".join(f"{k}: {100 * v:.2f}" for k, v in t["mean"].items()))

    summary = pd.concat({"overall": overall.stack(), "by_m": by_m["mean"]}, names=["table"])
    summary.to_csv(RESULTS / "undetectable_census.csv")

    fig, (ax_m, ax_pairs) = plt.subplots(1, 2, figsize=(13, 4.8))
    for set_name, color in (("minimal", "#2a78d6"), ("30 clues", "#eb6834")):
        t = by_m.loc[set_name]
        ax_m.plot(range(len(t)), t["mean"], "o-", color=color, linewidth=2, label=f"{set_name} puzzles")
        ax_m.set_xticks(range(len(t)), t.index)
    ax_m.axhline(100 / 8, color=MUTED, linewidth=1, linestyle="--")
    ax_m.text(0.1, 100 / 8 + 0.4, "1/8: the lemma's value at m = 2", color=MUTED, fontsize=9)
    style(ax_m)
    ax_m.set_xlabel("m = number of solutions once the clue is removed", color=INK)
    ax_m.set_ylabel("undetectable substitutions (%)", color=INK)
    ax_m.legend(frameon=False, labelcolor=INK)
    grid = np.full((9, 9), np.nan)
    for (t_digit, r_digit), v in rate.items():
        grid[t_digit - 1, r_digit - 1] = 100 * v
    im = ax_pairs.imshow(grid, cmap="Blues", vmin=0)
    ax_pairs.set_xticks(range(9), range(1, 10))
    ax_pairs.set_yticks(range(9), range(1, 10))
    ax_pairs.set_xlabel("digit read", color=INK)
    ax_pairs.set_ylabel("true digit", color=INK)
    ax_pairs.set_title(f"Minimal puzzles: undetectable rate by digit pair (%)\nχ² test of uniformity: p = {p:.2f}",
                       color=INK, fontsize=11)
    fig.colorbar(im, ax=ax_pairs)
    save(fig, RESULTS / "undetectable_census.png")


def run_margins(split: str, method: str, fits: dict) -> pd.DataFrame:
    """Return the answer status and margin of every grid of a split for every model, with the probabilities of
    the chosen calibration; cached in data/ so that the test grids run only once."""
    from calibrate import apply
    from read_grids import grid_logits

    path = DATA_DIR / f"margins_{split}_{method}.csv"
    if path.exists():
        return pd.read_csv(path)
    puzzles = load_puzzles(split)
    rows = []
    with Pool(WORKERS) as pool:
        for (name, seed), z in grid_logits(split).items():
            start = time.perf_counter()
            lp = torch.log_softmax(apply(z, fits[name, seed], method).double(), dim=2).tolist()
            results = pool.starmap(margin, [(lp[i], p["solution"]) for i, p in enumerate(puzzles)], chunksize=4)
            rows += [dict(model=name, seed=seed, grid=i, **r) for i, r in enumerate(results)]
            print(f"margins {split} {name} seed {seed}: {time.perf_counter() - start:.0f} s", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False)
    return table


def choose_tau(table: pd.DataFrame) -> float:
    """Return the threshold on the grid minimizing R * (wrong answers not flagged) + (right answers flagged),
    pooled over all models; the smallest one on ties."""
    wrong = table.margin[table.answer == "wrong"].to_numpy()
    right = table.margin[table.answer == "right"].to_numpy()
    costs = [R * (wrong >= t).sum() + (right < t).sum() for t in TAU_GRID]
    return float(TAU_GRID[int(np.argmin(costs))])


def margin_summary(table: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Return per model (seeds pooled): the answers by status, and what the margin rule and the next-reading rule flag."""
    rows = []
    for name in MODELS:
        t = table[table.model == name]
        right, wrong = t[t.answer == "right"], t[t.answer == "wrong"]
        caught, lost = (wrong.margin < tau).sum(), (right.margin < tau).sum()
        rows.append(dict(model=name, tau=tau, grids=len(t), right=len(right), wrong=len(wrong),
                         timeout=(t.answer == "timeout").sum(), flagged=(t.answer == "flagged").sum(),
                         wrong_caught=caught, right_flagged=lost,
                         right_kept=len(right) - lost, wrong_undetected=len(wrong) - caught,
                         next_wrong_caught=wrong.next_flag.astype(bool).sum(),
                         next_right_flagged=right.next_flag.astype(bool).sum(),
                         wrong_rival_is_truth=wrong.rival_is_truth.astype(bool).sum(),
                         margin_lower_bounds=(~t.margin_exact[t.answer.isin(["right", "wrong"])].astype(bool)).sum()))
    return pd.DataFrame(rows)


def plot_margins(table: pd.DataFrame, tau: float, split: str, curve: bool) -> None:
    """Save the margins of right and wrong answers per model, and (for the choice data) the trade-off curve."""
    fig, axes = plt.subplots(1, 2 if curve else 1, figsize=(13 if curve else 7, 4.8))
    ax_strip = axes[0] if curve else axes
    rng = np.random.default_rng(0)
    for row, name in enumerate(MODELS):
        t = table[table.model == name]
        for answer, color, size in (("right", MUTED, 6), ("wrong", BAD, 24)):
            m = t[t.answer == answer].margin.clip(upper=40)
            ax_strip.scatter(m, row + rng.uniform(-0.25, 0.25, len(m)), s=size, color=color, linewidths=0,
                             alpha=0.5 if answer == "right" else 0.9, label=f"{answer} answer" if row == 0 else None)
    ax_strip.axvline(tau, color=INK, linewidth=1, linestyle="--")
    ax_strip.text(tau + 0.3, 2.45, f"τ = {tau:.2f}", color=INK, fontsize=9)
    ax_strip.set_yticks(range(len(MODELS)), MODELS)
    ax_strip.set_xlabel("margin to the best competing answer (nats, clipped at 40)", color=INK)
    ax_strip.set_title(f"{split} grids, 3 seeds pooled", color=INK, fontsize=11)
    style(ax_strip)
    ax_strip.legend(frameon=False, labelcolor=INK, loc="lower right")
    if curve:
        ax_curve = axes[1]
        for name in MODELS:
            t = table[table.model == name]
            wrong, right = t.margin[t.answer == "wrong"], t.margin[t.answer == "right"]
            if len(wrong) == 0:
                continue
            ax_curve.plot([100 * (right < x).mean() for x in TAU_GRID], [100 * (wrong < x).mean() for x in TAU_GRID],
                          color=MODEL_COLORS[name], linewidth=2, label=name)
            ax_curve.scatter([100 * (right < tau).mean()], [100 * (wrong < tau).mean()], color=MODEL_COLORS[name],
                             s=60, edgecolors="white", linewidths=1.5, zorder=3)
        style(ax_curve)
        ax_curve.set_xscale("symlog", linthresh=1)
        ax_curve.set_xlabel("right answers flagged (%)", color=INK)
        ax_curve.set_ylabel("wrong answers caught (%)", color=INK)
        ax_curve.set_title(f"Trade-off as the threshold varies (dots: chosen τ)", color=INK, fontsize=11)
        ax_curve.legend(frameon=False, labelcolor=INK)
    save(fig, RESULTS / f"undetectable_margin_{split}.png")


def main() -> None:
    """Run the census, choose the threshold on the validation grids, then run the test grids once."""
    start = time.perf_counter()
    analyze_census(run_census())
    print(f"(census: {time.perf_counter() - start:.0f} s)\n")
    fits = torch.load(DATA_DIR / "calibration_fits.pt")
    method = pd.read_csv(RESULTS / "calibration_summary.csv").chosen.iloc[0]
    val = run_margins("val", method, fits)
    tau = choose_tau(val)
    print(f"Probabilities: {method} (chosen by calibrate.py). Threshold chosen on the validation grids: tau = {tau:.2f} "
          f"nats (ln R = {math.log(R):.2f}, the Bayes threshold for calibrated margins)")
    for split, table in (("val", val), ("test", run_margins("test", method, fits))):
        summary = margin_summary(table, tau)
        summary.to_csv(RESULTS / f"undetectable_margin_{split}.csv", index=False)
        print(f"\n{split} grids, 3 seeds pooled:\n" + summary.to_string(index=False))
        plot_margins(table, tau, split, curve=split == "val")


if __name__ == "__main__":
    main()
