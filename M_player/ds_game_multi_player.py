#!/usr/bin/env python3
"""Evolutionary simulation of the M-player resource game (original model).

Python counterpart of ds_game_multi_player.cpp: same model, same parameters,
same CSV output. The game is vectorised over groups with NumPy
(about 45 s per run at N = 100, M = 5, 3000 generations). This script also
draws the figures, either from its own runs or from the CSV files written by
the C++ program (--plot-from).

Model
-----
A population of N groups of M players. Every generation, each group plays a
game of T steps on its own resource x.

Game (one group, one step t)
    z_i(t)   = x(t) + S_i M y_i(t) + A_i M <y(t)>   <y>: group mean incl. self
    h_i(t)   = 1 if z_i(t) > 0 else 0              harvest or not
    H(t)     = sum_i h_i(t)
    r(x)     = x + alpha (x - x^2)                 alpha = 1
    harvest  = min(beta H, 1) r(x)                 beta  = 0.9 / M
    x(t+1)   = r(x) - harvest
    p(t)     = harvest / H                         (0 if H = 0)
    y_i(t+1) = (1 - kappa) y_i(t) + p(t) h_i(t)    kappa = 0.25

Initial state (every generation)
    x(0) = 0.1,   y_i(0) = (1 + eta_i) / M,   eta_i ~ N(0, 0.1)

Fitness
    f_i = mean of y_i(t) over t in [0.1 T, T)      (first 10 % discarded)

Reproduction
    Exactly N M offspring; each offspring draws its parent with probability
    proportional to f_i (multinomial / Wright-Fisher sampling). Offspring
    inherit (S, A) plus independent N(0, mu) mutations, mu = 0.1.
    Players are reassigned to groups at random every generation.

Initial population
    S_i, A_i ~ N(0, 0.1)

Total fitness Q = M x (mean fitness per player).

Usage
-----
    python ds_game_multi_player.py                        # N = 100, M = 5, trial 0
    python ds_game_multi_player.py --N 100 --M 5 --trials 10
    python ds_game_multi_player.py --gens 500 --steps 500 # quick test
    python ds_game_multi_player.py --plot-from Mplayer_out --N 100 --M 5 --trials 10
                                        # figures from the C++ output

Output (in <out>/res/ and <out>/figs/)
    gen_N{N}_M{M}_trial{t}.csv   one row per generation: Q, mean fitness,
                                 harvesting frequency, mean resource,
                                 mean/sd of S and A, median S/A, fraction of
                                 players with -1.4 < S/A < -1.0
    snap_N{N}_M{M}_trial{t}.csv  recorded generations: S, A, y_i(0), fitness
                                 and harvest rate of every player
    dyn_N{N}_M{M}_trial{t}.csv   recorded generations: resource, y and h of
                                 groups 0-2 over the last 300 game steps
    summary_N{N}_M{M}.csv        one row per run: averages over the last
                                 500 generations
    figs/*_traj.pdf              mean S, A and Q over generations
    figs/*_SA.pdf                (S, A) of every player, recorded generations
    figs/*_gen{g}_gamesteps.pdf  actions and resource of group 0

A group of a recorded generation can be replayed exactly by passing the
(S, A, y0) columns of snap_*.csv to `play_generation(S, A, rng, y0=y0)`.
The random number streams of the Python and C++ versions differ, so single
runs are not identical between them, but their statistics are.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

# ==================================================
# parameters (identical to ds_game_multi_player.cpp)
# ==================================================
NUM_STEPS = 1000        # T, game steps per generation
NUM_GENERATIONS = 3000
ALPHA = 1.0             # resource growth rate
KAPPA = 0.25            # decay rate of y
BETA_M = 0.9            # beta = BETA_M / M
MUTATION = 0.1          # sd of the mutation of S and A
X0 = 0.1                # x(0) in every generation
TRANSIENT = 0.1         # fraction of T excluded from fitness
INIT_SD = 0.1           # sd of S and A in generation 0
TAIL_GENS = 500         # window of the summary averages

# recorded generations (1-based labels; the last generation is always added)
DETAIL_GENS = [100, 300, 1000, 3000]
DETAIL_STEPS = 300      # last steps stored per recorded generation
DETAIL_GROUPS = 3       # groups stored per recorded generation

# reference band of the evolved decision rule: -1.4 < S/A < -1.0
BAND_LO, BAND_HI = -1.4, -1.0

# player colours (Okabe-Ito, extended), assigned in order of fitness rank
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
           "#56B4E9", "#F0E442", "#7F3C8D", "#8C6D31", "#666666"]

GEN_COLS = ["generation", "Q", "mean_fitness", "sd_fitness", "mean_action",
            "mean_resource", "mean_S", "sd_S", "mean_A", "sd_A",
            "ratio_med", "band_frac"]


# ==================================================
# one generation of the game (no reproduction)
# ==================================================
def play_generation(S, A, rng, num_steps=NUM_STEPS, y0=None,
                    record_groups=0, record_steps=DETAIL_STEPS):
    """Play one generation in all groups at once.

    Parameters
    ----------
    S, A : ndarray, shape (N, M)
        Strategies of the player in seat k of group g.
    rng : np.random.Generator
    y0 : None or ndarray, shape (N, M)
        Initial richness y_i(0). None draws (1 + N(0, 0.1)) / M.
    record_groups : int
        If > 0, store the last `record_steps` steps of groups
        0 ... record_groups - 1.

    Returns
    -------
    dict with
        fit (N, M)           fitness
        y0 (N, M)            y_i(0) used
        harvest_rate (N, M)  fraction of steps in which the player harvested
        mean_action          mean of h over all players and steps
        mean_resource        mean of x(t+1) over all groups and steps
        trace                if record_groups > 0: x (L, G), y and h (L, G, M),
                             values before the update of each step, and step0
    """
    N, M = S.shape
    beta = BETA_M / M
    keep = 1.0 - KAPPA
    fit_from = min(max(0, int(round(TRANSIENT * num_steps))), num_steps - 1)

    if y0 is None:
        y0 = (1.0 + rng.normal(0.0, 0.1, size=(N, M))) * (1.0 / M)
    y = np.array(y0, dtype=float)
    x = np.full(N, X0)
    SM = S * M

    fit_sum = np.zeros((N, M))
    h_count = np.zeros((N, M))
    action_sum = 0.0
    resource_sum = 0.0

    trace = None
    rec_from = max(0, num_steps - record_steps)
    if record_groups > 0:
        G = min(record_groups, N)
        L = num_steps - rec_from
        trace = dict(x=np.empty((L, G)), y=np.empty((L, G, M)),
                     h=np.empty((L, G, M)), step0=rec_from)

    for t in range(num_steps):
        # M <y>, in the same order of operations as the C++ code
        peer = M * y.sum(axis=1, keepdims=True) / M
        z = x[:, None] + SM * y + A * peer
        h = (z > 0.0).astype(float)
        H = h.sum(axis=1)

        grown = x + ALPHA * (x - x * x)
        harvested = np.minimum(beta * H, 1.0) * grown
        per_unit = np.divide(harvested, H, out=np.zeros(N), where=H > 0)

        if t >= fit_from:
            fit_sum += y                   # y(t) before the update
        h_count += h
        if trace is not None and t >= rec_from:
            i = t - rec_from
            trace["x"][i] = x[:G]
            trace["y"][i] = y[:G]
            trace["h"][i] = h[:G]

        x = grown - harvested
        y = keep * y + per_unit[:, None] * h
        action_sum += H.sum()
        resource_sum += x.sum()

    fit = fit_sum / (num_steps - fit_from)
    fit[~(np.isfinite(fit) & (fit > 0.0))] = 0.0

    return dict(fit=fit, y0=np.asarray(y0), harvest_rate=h_count / num_steps,
                mean_action=action_sum / (num_steps * N * M),
                mean_resource=resource_sum / (num_steps * N),
                trace=trace)


# ==================================================
# reproduction: exactly n offspring, parents drawn in proportion to fitness
# ==================================================
def reproduce(S, A, fit, rng, mutation=MUTATION):
    """S, A, fit: 1-D arrays of length n. Returns the offspring's (S, A)."""
    n = S.size
    total = fit.sum()
    if total > 0.0 and np.isfinite(total):
        parents = rng.choice(n, size=n, p=fit / total)
    else:                              # every fitness is zero: no selection
        parents = rng.integers(0, n, size=n)
    S_kid = S[parents] + rng.normal(0.0, mutation, n)
    A_kid = A[parents] + rng.normal(0.0, mutation, n)
    return S_kid, A_kid


# ==================================================
# one run
# ==================================================
def make_rng(N, M, trial, seed=0):
    return np.random.default_rng([seed, N, M, trial])


def run_trial(N=100, M=5, trial=0, num_generations=NUM_GENERATIONS,
              num_steps=NUM_STEPS, detail_gens=DETAIL_GENS,
              detail_groups=DETAIL_GROUPS, seed=0, verbose=True):
    rng = make_rng(N, M, trial, seed)
    n = N * M
    S = rng.normal(0.0, INIT_SD, n)
    A = rng.normal(0.0, INIT_SD, n)

    # recorded generations: 0-based index -> 1-based label
    rec = {}
    for g in detail_gens:
        lab = min(max(1, g), num_generations)
        rec[lab - 1] = lab
    rec[num_generations - 1] = num_generations

    gen_rows = np.empty((num_generations, len(GEN_COLS)))
    snaps = {}

    for gen in range(num_generations):
        perm = rng.permutation(n)      # random regrouping
        S, A = S[perm], A[perm]
        is_rec = gen in rec
        out = play_generation(S.reshape(N, M), A.reshape(N, M), rng,
                              num_steps=num_steps,
                              record_groups=detail_groups if is_rec else 0)
        fit = out["fit"].ravel()

        nz = np.abs(A) > 1e-12
        ratio = S[nz] / A[nz]
        mf = fit.mean()
        gen_rows[gen] = [gen, M * mf, mf, fit.std(), out["mean_action"],
                         out["mean_resource"], S.mean(), S.std(), A.mean(),
                         A.std(),
                         np.median(ratio) if ratio.size else np.nan,
                         np.count_nonzero((ratio > BAND_LO) & (ratio < BAND_HI))
                         / n]

        if is_rec:
            snaps[rec[gen]] = dict(generation=gen, S=S.copy(), A=A.copy(),
                                   y0=out["y0"].ravel().copy(), fit=fit.copy(),
                                   harvest_rate=out["harvest_rate"].ravel(),
                                   trace=out["trace"])
        if verbose and (gen % 500 == 0 or gen == num_generations - 1):
            print(f"  N={N} M={M} trial={trial} gen={gen:5d}  "
                  f"Q={M * mf:.4f}  S={S.mean():+.3f}  A={A.mean():+.3f}")

        S, A = reproduce(S, A, fit, rng)

    gen = dict(zip(GEN_COLS, gen_rows.T))
    return dict(N=N, M=M, trial=trial, gen=gen, snaps=snaps,
                summary=summarise(gen, N, M, trial))


def summarise(gen, N, M, trial):
    ng = len(gen["generation"])
    tail = slice(max(0, ng - TAIL_GENS), ng)
    r = gen["ratio_med"][tail]
    r = r[np.isfinite(r)]
    return dict(N=N, M=M, trial=trial,
                Q_tail=gen["Q"][tail].mean(),
                S_tail=gen["mean_S"][tail].mean(),
                A_tail=gen["mean_A"][tail].mean(),
                ratio_med_tail=np.median(r) if r.size else np.nan,
                band_tail=gen["band_frac"][tail].mean(),
                action_tail=gen["mean_action"][tail].mean(),
                resource_tail=gen["mean_resource"][tail].mean())


# ==================================================
# CSV input / output
# ==================================================
def tag_of(N, M, trial):
    return f"N{N}_M{M}_trial{trial}"


def save_csv(res, res_dir):
    os.makedirs(res_dir, exist_ok=True)
    N, M, trial = res["N"], res["M"], res["trial"]
    tag = tag_of(N, M, trial)

    g = res["gen"]
    with open(os.path.join(res_dir, f"gen_{tag}.csv"), "w") as f:
        f.write(",".join(GEN_COLS) + "\n")
        for i in range(len(g["generation"])):
            f.write(f"{int(g['generation'][i])},"
                    + ",".join(f"{g[c][i]:.8g}" for c in GEN_COLS[1:]) + "\n")

    with open(os.path.join(res_dir, f"snap_{tag}.csv"), "w") as f:
        f.write("generation,label,M,index,group,seat,S,A,y0,fitness,"
                "harvest_rate\n")
        for label, s in sorted(res["snaps"].items()):
            for i in range(s["S"].size):
                f.write(f"{s['generation']},{label},{M},{i},{i // M},{i % M},"
                        f"{s['S'][i]:.10g},{s['A'][i]:.10g},{s['y0'][i]:.10g},"
                        f"{s['fit'][i]:.10g},{s['harvest_rate'][i]:.10g}\n")

    with open(os.path.join(res_dir, f"dyn_{tag}.csv"), "w") as f:
        f.write("generation,label,group,step,resource,seat,y,h\n")
        for label, s in sorted(res["snaps"].items()):
            tr = s["trace"]
            L, G, _ = tr["y"].shape
            for gi in range(G):
                for t in range(L):
                    for k in range(M):
                        f.write(f"{s['generation']},{label},{gi},"
                                f"{tr['step0'] + t},{tr['x'][t, gi]:.10g},{k},"
                                f"{tr['y'][t, gi, k]:.10g},"
                                f"{int(tr['h'][t, gi, k])}\n")

    path = os.path.join(res_dir, f"summary_N{N}_M{M}.csv")
    new = not os.path.exists(path)
    sm = res["summary"]
    with open(path, "a") as f:
        if new:
            f.write(",".join(sm.keys()) + "\n")
        f.write(",".join(f"{v:.8g}" if isinstance(v, float) else str(v)
                         for v in sm.values()) + "\n")


def _read_csv(path):
    """Read a numeric CSV with a header into a dict of 1-D arrays."""
    with open(path) as f:
        cols = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return {c: data[:, i] for i, c in enumerate(cols)}


def load_result(res_dir, N, M, trial):
    """Rebuild the result of one run from its CSV files (Python or C++)."""
    tag = tag_of(N, M, trial)
    gen = _read_csv(os.path.join(res_dir, f"gen_{tag}.csv"))
    snaps = {}
    snap_path = os.path.join(res_dir, f"snap_{tag}.csv")
    dyn_path = os.path.join(res_dir, f"dyn_{tag}.csv")
    if os.path.exists(snap_path) and os.path.exists(dyn_path):
        sn = _read_csv(snap_path)
        dy = _read_csv(dyn_path)
        for lab in np.unique(sn["label"]).astype(int):
            m = sn["label"] == lab
            order = np.argsort(sn["index"][m])
            d = dy["label"] == lab
            G = int(dy["group"][d].max()) + 1
            steps = np.unique(dy["step"][d]).astype(int)
            L, step0 = steps.size, int(steps[0])
            ti = dy["step"][d].astype(int) - step0
            gi = dy["group"][d].astype(int)
            ki = dy["seat"][d].astype(int)
            tr = dict(x=np.empty((L, G)), y=np.empty((L, G, M)),
                      h=np.empty((L, G, M)), step0=step0)
            tr["x"][ti, gi] = dy["resource"][d]
            tr["y"][ti, gi, ki] = dy["y"][d]
            tr["h"][ti, gi, ki] = dy["h"][d]
            snaps[int(lab)] = dict(
                generation=int(sn["generation"][m][0]),
                S=sn["S"][m][order], A=sn["A"][m][order],
                y0=sn["y0"][m][order], fit=sn["fitness"][m][order],
                harvest_rate=sn["harvest_rate"][m][order], trace=tr)
    return dict(N=N, M=M, trial=trial, gen=gen, snaps=snaps,
                summary=summarise(gen, N, M, trial))


# ==================================================
# figures (no in-figure titles; the condition is in the file name)
# ==================================================
def plot_trajectory(res, filepath):
    """Left axis: mean S and A. Right axis: total fitness Q."""
    import matplotlib.pyplot as plt
    g = res["gen"]
    t = g["generation"]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    l1 = ax1.plot(t, g["mean_S"], "-", color=PALETTE[0], lw=1.5, label="mean S")
    l2 = ax1.plot(t, g["mean_A"], "--", color=PALETTE[1], lw=1.5,
                  label="mean A")
    ax1.axhline(0.0, color="0.7", lw=0.6)
    ax1.set_xlabel("generation")
    ax1.set_ylabel("strategy (S, A)")
    ax2 = ax1.twinx()
    l3 = ax2.plot(t, g["Q"], "-", color="0.6", lw=0.8, label="Q")
    ax2.set_ylabel("total fitness Q")
    ax1.set_zorder(ax2.get_zorder() + 1)   # draw S and A on top of Q
    ax1.patch.set_visible(False)
    lines = l1 + l2 + l3
    ax1.legend(lines, [l.get_label() for l in lines], loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def plot_gamesteps(snap, M, filepath, group=0, window=40):
    """Game of one group in a recorded generation: harvest (filled) or not
    (open) for every player, and the resource x.

    Players are ordered by fitness, highest on top, and coloured by rank.
    """
    import matplotlib.pyplot as plt
    tr = snap["trace"]
    L = tr["x"].shape[0]
    w = min(window, L)
    steps = tr["step0"] + np.arange(L)[-w:]
    h = tr["h"][-w:, group, :]
    x = tr["x"][-w:, group]
    fit = snap["fit"][group * M:(group + 1) * M]
    order = np.argsort(-fit)

    fig, (ax_r, ax_x) = plt.subplots(
        2, 1, figsize=(7, 1.2 + 0.25 * M + 2.0), sharex=True,
        gridspec_kw=dict(height_ratios=[0.25 * M + 0.4, 2.0]))
    for rank, k in enumerate(order):
        c = PALETTE[rank % len(PALETTE)]
        on = h[:, k] > 0.5
        yv = np.full(w, M - 1 - rank)
        ax_r.scatter(steps[on], yv[on], marker="s", s=28, color=c)
        ax_r.scatter(steps[~on], yv[~on], marker="s", s=28, facecolors="none",
                     edgecolors=c)
    ax_r.set_yticks(range(M))
    ax_r.set_yticklabels([f"{M - i}" for i in range(M)])
    ax_r.set_ylabel("player\n(fitness rank)")
    ax_r.set_ylim(-0.7, M - 0.3)

    ax_x.plot(steps, x, "-", color="0.2", lw=1.5)
    ax_x.set_ylim(-0.02, 1.02)
    ax_x.set_xlabel("game step")
    ax_x.set_ylabel("resource x")
    ax_x.set_xlim(steps[0] - 0.5, steps[-1] + 0.5)
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_cloud(res, filepath):
    """(S, A) of every player in the recorded generations, coloured by
    generation. Dotted lines: S/A = -1.4 and -1.0."""
    import matplotlib.pyplot as plt
    snaps = res["snaps"]
    labels = sorted(snaps)
    cmap = plt.get_cmap("plasma")
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, lab in enumerate(labels):
        s = snaps[lab]
        ax.scatter(s["S"], s["A"], s=6, alpha=0.5, lw=0,
                   color=cmap(i / max(1, len(labels) - 1) * 0.9),
                   label=f"gen {lab}")
    lim = ax.get_xlim()
    ss = np.linspace(*lim, 2)
    for r in (BAND_LO, BAND_HI):
        ax.plot(ss, ss / r, ":", color="0.6", lw=0.8)
    ax.set_xlim(lim)
    ax.set_xlabel("S")
    ax.set_ylabel("A")
    ax.legend(fontsize=8, markerscale=2, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def save_figs(res, figs_dir):
    os.makedirs(figs_dir, exist_ok=True)
    M = res["M"]
    tag = tag_of(res["N"], M, res["trial"])
    plot_trajectory(res, os.path.join(figs_dir, f"{tag}_traj.pdf"))
    if res["snaps"]:
        plot_strategy_cloud(res, os.path.join(figs_dir, f"{tag}_SA.pdf"))
    for lab, s in sorted(res["snaps"].items()):
        plot_gamesteps(s, M, os.path.join(figs_dir,
                                          f"{tag}_gen{lab}_gamesteps.pdf"))


# ==================================================
# main
# ==================================================
def main():
    ap = argparse.ArgumentParser(
        description="Evolutionary simulation of the M-player resource game.")
    ap.add_argument("--N", type=int, default=100, help="number of groups")
    ap.add_argument("--M", type=int, default=5, help="players per group")
    ap.add_argument("--trials", type=int, default=1,
                    help="number of independent runs")
    ap.add_argument("--trial0", type=int, default=0,
                    help="index of the first run")
    ap.add_argument("--gens", type=int, default=NUM_GENERATIONS,
                    help="generations")
    ap.add_argument("--steps", type=int, default=NUM_STEPS,
                    help="game steps per generation, T")
    ap.add_argument("--seed", type=int, default=0,
                    help="extra seed, mixed into every run seed")
    ap.add_argument("--out", default="Mplayer_out", help="output directory")
    ap.add_argument("--no-figs", action="store_true", help="skip the figures")
    ap.add_argument("--plot-from", metavar="DIR", default=None,
                    help="do not simulate; draw the figures from DIR/res "
                         "(e.g. the output of the C++ program) into DIR/figs")
    args = ap.parse_args()

    trials = range(args.trial0, args.trial0 + args.trials)

    if args.plot_from is not None:
        for trial in trials:
            res = load_result(os.path.join(args.plot_from, "res"),
                              args.N, args.M, trial)
            save_figs(res, os.path.join(args.plot_from, "figs"))
            print(f"trial {trial}: figures written to "
                  f"{os.path.join(args.plot_from, 'figs')}")
        return

    res_dir = os.path.join(args.out, "res")
    figs_dir = os.path.join(args.out, "figs")
    for trial in trials:
        res = run_trial(N=args.N, M=args.M, trial=trial,
                        num_generations=args.gens, num_steps=args.steps,
                        seed=args.seed)
        save_csv(res, res_dir)
        if not args.no_figs:
            save_figs(res, figs_dir)
        sm = res["summary"]
        print(f"trial {trial}: Q_tail={sm['Q_tail']:.4f}  "
              f"S_tail={sm['S_tail']:+.3f}  A_tail={sm['A_tail']:+.3f}  "
              f"S/A={sm['ratio_med_tail']:+.3f}")


if __name__ == "__main__":
    main()
