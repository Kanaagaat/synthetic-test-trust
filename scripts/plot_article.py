"""Article figures for RQ1 and RQ2 from viz/rq1_many/summary.json (eval_many.py).

    .venv/bin/python scripts/plot_article.py      # -> viz/article/fig*_{ru,en}.{png,pdf} + fig*.csv

fig1_rq2_variants       RQ2: glioma metrics on the full real test per variant (3 seeds + mean, best.pt)
                        and seed-mean contrasts of AP50-95 with 95% group-bootstrap CI
fig2_rq1_scatter        RQ1: each model's AP50 on the real test vs the S-test (all RQ1 models), one panel
                        per training variant (3-column grid), other models in gray, y = x reference
fig2b_..._recall        the same for recall@0.25 (supplementary)
fig3_rq1_contrasts      RQ1 x RQ2: variant contrasts measured on the real test vs on the S-test (CI)
                        (main scenario "dataset boxes"; fig3b = one blind annotator)
fig4_rq1_pairs          RQ1: agreement of the two tests on every model pair (1128 pairs), both scenarios

    .venv/bin/python scripts/plot_article.py --paper   # -> viz/article/paper/fig*_en.pdf: no built-in
                        title/subtitle (the LaTeX caption carries them), white background, tight crop

Colors: reference categorical slots (validated with the dataviz validator, light surface #fcfcfb):
two series blue/orange pass all-pairs; the 4 pair categories blue/orange/aqua/violet pass adjacent
(aqua < 3:1 contrast -> segments carry printed counts, legend always present, numbers also in CSV).
Four variants on one scatter do not pass all-pairs, so the variants are small multiples instead.
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "viz" / "rq1_many" / "summary.json"
OUT = ROOT / "viz" / "article"

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, VIOLET, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#d6d5cf"
CONTEXT = "#cfcec8"  # other models in the small multiples

# variants and contrasts actually present in summary.json are picked in configure(), in this order
ORDER = ["A_full", "A_low", "B_low", "C_low", "C_full", "D"]
VARIANTS = ORDER[:4]
VNAME = {
    "ru": {"A_full": "A_full: все реальные", "A_low": "A_low: 25% глиом",
           "B_low": "B_low: A_low + копии", "C_low": "C_low: A_low + синтетика",
           "C_full": "C_full: A_full + синтетика", "D": "D: синтетика вместо глиом"},
    "en": {"A_full": "A_full: all real", "A_low": "A_low: 25% of gliomas",
           "B_low": "B_low: A_low + copies", "C_low": "C_low: A_low + synthetic",
           "C_full": "C_full: A_full + synthetic", "D": "D: synthetic gliomas only"},
}
MNAME = {"AP50": "AP50", "recall@0.25": "Recall (conf ≥ 0.25)", "AP50-95": "AP50-95"}
ALL_CONTRASTS = ["A_low - A_full", "C_low - A_full", "B_low - A_low", "C_low - A_low", "C_low - B_low",
                 "C_full - A_full", "D - A_full"]
F1_ORDER = ["A_low - A_full", "B_low - A_low", "C_low - A_low", "C_low - B_low", "C_low - A_full",
            "C_full - A_full", "D - A_full"]
CONTRASTS = ALL_CONTRASTS[:5]
COUNTS = {}
PAPER = False  # --paper: figures for the LaTeX manuscript (no header, white, tight bbox)


def configure(s):
    """Use the variants/contrasts that summary.json has; counts for the figure subtitles."""
    global VARIANTS, CONTRASTS
    have = s["rq2"]["by_variant"]
    VARIANTS = [v for v in ORDER if v in have]
    extra = sorted(set(have) - set(ORDER))
    assert not extra, f"no name/order for variants {extra}: add them to ORDER and VNAME"
    vc = s["rq1"]["dataset"]["variant_contrasts"]
    CONTRASTS = [c for c in ALL_CONTRASTS if c in vc]
    n = len(s["rq1_models"])
    COUNTS.update(n_models=n, n_runs=n // 4, n_pairs=n * (n - 1) // 2)

T = {
    "ru": {
        "f1_title": "Синтетика не заменяет реальные глиомы; прирост — слабый в low, нулевой в full",
        "f1_sub": "Реальный тест (203 снимка), класс «глиома», best.pt. Точки — 3 сида, черта — среднее. "
                  "Внизу: разница средних по сидам, AP50-95, 95% ДИ (2000 bootstrap-ресэмплов групп).",
        "f1_contr": "Разница AP50-95 (среднее по сидам)", "seed": "сид", "mean": "среднее",
        "f2_title": "S-test завышает качество почти всех моделей, и тем сильнее, чем слабее модель",
        "f2_sub": "{n_models} моделей ({n_runs} прогонов × чекпоинты 25/50/75 + best), рамки датасета, 51 пара «реальный снимок — двойник». "
                  "Над диагональю — S-test оценивает модель выше реального теста.",
        "real": "Реальный тест", "stest": "S-test (синтетические двойники)", "diag": "y = x",
        "this": "модели варианта", "others": "остальные модели", "best": "best.pt",
        "f3_title": "S-test не замечает ни нехватки реальных данных, ни замены глиом синтетикой",
        "f3_sub": "Разница вариантов (среднее по 3 сидам, best.pt), измеренная на реальном тесте и на S-test. "
                  "Линии — 95% ДИ, bootstrap по группам.",
        "f3b_sub": "То же при одном разметчике вслепую (45 пар). "
                   "Разница вариантов (среднее по 3 сидам, best.pt), линии — 95% ДИ.",
        "diff": "Разница метрики", "zero": "нет разницы",
        "f4_title": "Где S-test уверен, реальный тест часто не может различить модели",
        "f4_sub": "Все пары из {n_models} моделей ({n_pairs}). «Уверен» — 95% ДИ разницы пары не содержит 0. "
                  "Справа — τ Кендалла между рейтингами двух тестов.",
        "p_both": "оба теста уверены, порядок один", "p_syn": "уверен только S-test",
        "p_real": "уверен только реальный тест", "p_opp": "оба уверены, порядок противоположный",
        "p_none": "ни один не уверен", "sc_dataset": "рамки датасета", "sc_same": "один разметчик",
        "share": "Доля пар моделей", "opp_short": "противоположный порядок",
    },
    "en": {
        "f1_title": "Synthetic images do not replace real gliomas; small gain in low data, none in full",
        "f1_sub": "Real test (203 images), glioma class, best.pt. Dots: 3 seeds; bar: mean. "
                  "Bottom: difference of seed means, AP50-95, 95% CI (2000 group-bootstrap resamples).",
        "f1_contr": "AP50-95 difference (seed means)", "seed": "seed", "mean": "mean",
        "f2_title": "The S-test overstates almost every model, and the weaker the model, the more",
        "f2_sub": "{n_models} models ({n_runs} runs × checkpoints 25/50/75 + best), dataset boxes, 51 real-twin pairs. "
                  "Above the diagonal the S-test rates the model higher than the real test.",
        "real": "Real test", "stest": "S-test (synthetic twins)", "diag": "y = x",
        "this": "models of this variant", "others": "other models", "best": "best.pt",
        "f3_title": "The S-test misses both scarce real data and real gliomas replaced by synthetic ones",
        "f3_sub": "Variant differences (mean of 3 seeds, best.pt) measured on the real test and on the S-test. "
                  "Lines: 95% CI, group bootstrap.",
        "f3b_sub": "Same with one blind annotator (45 pairs). "
                   "Variant differences (mean of 3 seeds, best.pt); lines: 95% CI.",
        "diff": "Metric difference", "zero": "no difference",
        "f4_title": "Where the S-test is confident, the real test often cannot separate the models",
        "f4_sub": "All pairs of the {n_models} models ({n_pairs}). Confident: the 95% CI of the pair difference excludes 0. "
                  "Right: Kendall τ between the two tests' rankings.",
        "p_both": "both confident, same order", "p_syn": "only the S-test confident",
        "p_real": "only the real test confident", "p_opp": "both confident, opposite order",
        "p_none": "neither confident", "sc_dataset": "dataset boxes", "sc_same": "one annotator",
        "share": "Share of model pairs", "opp_short": "opposite order",
    },
}


def fmt(v, nd=2):
    v = 0.0 if abs(v) < 0.5 * 10 ** -nd else v
    return f"{v:+.{nd}f}".replace("-", "−")


def style(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    if grid_axis:
        getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def header(fig, title, sub, y=0.985):
    if PAPER:
        return
    fig.suptitle(title, x=0.02, y=y, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.02, y - 0.05, sub, ha="left", va="top", fontsize=9.3, color=INK2, wrap=True)


def save(fig, name, lang):
    if PAPER:
        out = OUT / "paper"
        out.mkdir(parents=True, exist_ok=True)
        fig.patch.set_facecolor("white")
        for ax in fig.axes:
            ax.set_facecolor("white")
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}_{lang}.{ext}", facecolor="white", dpi=200, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}_{lang}.{ext}", facecolor=SURFACE, dpi=200)
    plt.close(fig)


def contrast_label(c, lang):
    a, b = c.split(" - ")
    return f"{a} − {b}"


# ---------------------------------------------------------------- fig 1: RQ2
def fig1(s, lang):
    t, rq2 = T[lang], s["rq2"]
    metrics = ["AP50-95", "AP50", "recall@0.25"]
    fig = plt.figure(figsize=(12, 7.4 + 0.45 * (len(VARIANTS) - 4)))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.32, wspace=0.16,
                          left=0.205, right=0.97, top=0.80, bottom=0.08)
    ys = {v: -i for i, v in enumerate(VARIANTS)}
    for j, m in enumerate(metrics):
        ax = fig.add_subplot(gs[0, j])
        style(ax)
        for v in VARIANTS:
            bv = rq2["by_variant"][v][m]
            vals = bv["values"]
            ax.plot(vals, [ys[v]] * len(vals), "o", ms=8, mfc=BLUE, mec=SURFACE, mew=1.5, alpha=0.9, zorder=3,
                    label=t["seed"] if (j == 0 and v == VARIANTS[0]) else None)
            ax.plot([bv["mean"]] * 2, [ys[v] - 0.28, ys[v] + 0.28], color=INK, lw=2.2, solid_capstyle="round",
                    zorder=4, label=t["mean"] if (j == 0 and v == VARIANTS[0]) else None)
            ax.text(bv["mean"], ys[v] + 0.33, f"{bv['mean']:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        lo = min(min(rq2["by_variant"][v][m]["values"]) for v in VARIANTS)
        hi = max(max(rq2["by_variant"][v][m]["values"]) for v in VARIANTS)
        pad = (hi - lo) * 0.12
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-len(VARIANTS) + 0.45, 0.75)
        ax.set_yticks([ys[v] for v in VARIANTS], [VNAME[lang][v] for v in VARIANTS] if j == 0 else [])
        ax.tick_params(axis="y", labelsize=10, labelcolor=INK, pad=8)
        ax.set_title(MNAME[m], loc="left", fontsize=11, fontweight="bold", color=INK, pad=4)

    ax = fig.add_subplot(gs[1, :])
    style(ax)
    cs = rq2["contrasts_seed_mean"]
    order = [c for c in F1_ORDER if c in cs]
    for i, c in enumerate(order):
        d, (lo, hi) = cs[c]["AP50-95_mean_diff"], cs[c]["ci95"]
        y = -i
        ax.plot([lo, hi], [y, y], color=INK2, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(d, y, "o", ms=9, mfc=INK, mec=SURFACE, mew=2, zorder=3)
        ax.text(hi + 0.006, y, f"{fmt(d, 3)}  [{fmt(lo, 3)}, {fmt(hi, 3)}]", va="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    ax.text(0.002, 1.0, t["zero"], color=INK2, fontsize=8.5, va="bottom", transform=ax.get_xaxis_transform())
    ax.set_yticks([-i for i in range(len(order))], [contrast_label(c, lang) for c in order])
    ax.tick_params(axis="y", labelsize=10, labelcolor=INK, pad=8)
    ax.set_ylim(-len(order) + 0.4, 0.6)
    xlo = min(cs[c]["ci95"][0] for c in order)
    xhi = max(cs[c]["ci95"][1] for c in order)
    ax.set_xlim(min(-0.3, xlo - 0.02), max(0.2, xhi + 0.085))  # room for the printed CI on the right
    ax.set_xlabel(t["f1_contr"], color=INK2, fontsize=10)
    fig.legend(loc="upper left", bbox_to_anchor=(0.02, 0.875), ncol=2, frameon=False, fontsize=9,
               labelcolor=INK, handletextpad=0.4)
    header(fig, t["f1_title"], t["f1_sub"])
    save(fig, "fig1_rq2_variants", lang)


def fig1_csv(s):
    rq2 = s["rq2"]
    with open(OUT / "fig1_rq2_variants.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "AP50-95", "AP50-95_ci_lo", "AP50-95_ci_hi", "AP50", "recall@0.25", "bg_fp_rate@0.25"])
        for name, r in rq2["per_run"].items():
            w.writerow([name, round(r["AP50-95"], 4), round(r["AP50-95_ci95"][0], 4), round(r["AP50-95_ci95"][1], 4),
                        round(r["AP50"], 4), round(r["recall@0.25"], 4), round(r["bg_fp_rate@0.25"], 4)])
        w.writerow([])
        w.writerow(["contrast (seed means)", "AP50-95_diff", "ci_lo", "ci_hi"])
        for c, v in rq2["contrasts_seed_mean"].items():
            w.writerow([c, round(v["AP50-95_mean_diff"], 4), round(v["ci95"][0], 4), round(v["ci95"][1], 4)])


# ---------------------------------------------------------------- fig 2: RQ1 scatter
def fig2(s, lang, m="AP50", name="fig2_rq1_scatter"):
    """One metric per figure; variants on a grid (3 columns when there are more than 4)."""
    t = T[lang]
    pm = s["rq1"]["dataset"]["per_model"]
    models = s["rq1_models"]
    nv = len(VARIANTS)
    ncol = 3 if nv > 4 else nv
    nrow = -(-nv // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol + 0.6, 3.25 * nrow + 1.5), sharex=True, sharey=True,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    allx = [pm[n]["real"][m] for n in models]
    ally = [pm[n]["s_test"][m] for n in models]
    lo = min(min(allx), min(ally)) - 0.03
    for k, v in enumerate(VARIANTS):
        i, j = divmod(k, ncol)
        ax = axes[i, j]
        style(ax, grid_axis=None)
        ax.grid(True, color=GRID, lw=0.8)
        ax.plot([0, 1.02], [0, 1.02], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        oth = [n for n in models if not n.startswith(v + "_")]
        ax.plot([pm[n]["real"][m] for n in oth], [pm[n]["s_test"][m] for n in oth], "o", ms=5,
                color=CONTEXT, mec=SURFACE, mew=0.8, zorder=2, label=t["others"])
        mine = [n for n in models if n.startswith(v + "_")]
        ck = [n for n in mine if not n.endswith("@best")]
        bt = [n for n in mine if n.endswith("@best")]
        ax.plot([pm[n]["real"][m] for n in ck], [pm[n]["s_test"][m] for n in ck], "o", ms=7,
                mfc=BLUE, mec=SURFACE, mew=1.5, alpha=0.75, zorder=3, label=t["this"] + " (25/50/75)")
        ax.plot([pm[n]["real"][m] for n in bt], [pm[n]["s_test"][m] for n in bt], "D", ms=8,
                mfc=BLUE, mec=INK, mew=1.2, zorder=4, label=t["this"] + f" ({t['best']})")
        ax.set_xlim(lo, 1.025)
        ax.set_ylim(lo, 1.025)
        for ln in ax.lines[1:]:  # markers only; the y = x line (first) stays clipped to the panel
            ln.set_clip_on(False)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(VNAME[lang][v], fontsize=10.5, fontweight="bold", color=INK, loc="left")
        if j == 0:
            ax.set_ylabel(f"{MNAME[m]}\n{t['stest']}", color=INK2, fontsize=9.5)
        if i == nrow - 1 or k + ncol >= nv:
            ax.set_xlabel(f"{MNAME[m]}, {t['real'].lower()}", color=INK2, fontsize=9)
            ax.xaxis.set_tick_params(labelbottom=True)
        if k == 0:
            ax.text(lo + 0.02, lo + 0.05, t["diag"], color=INK2, fontsize=8.5, rotation=45,
                    rotation_mode="anchor", transform_rotates_text=True)
    for k in range(nv, nrow * ncol):
        axes[divmod(k, ncol)].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    top = 1 - 1.45 / fig.get_figheight()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, top + 0.075), ncol=3, frameon=False, fontsize=9,
               labelcolor=INK, handletextpad=0.3, columnspacing=1.5)
    fig.subplots_adjust(left=0.1, right=0.98, top=top, bottom=0.07, wspace=0.08, hspace=0.3)
    header(fig, t["f2_title"], t["f2_sub"].format(**COUNTS), y=0.985)
    save(fig, name, lang)


def fig2_csv(s):
    with open(OUT / "fig2_rq1_scatter.csv", "w", newline="") as f:
        w = csv.writer(f)
        mets = ["AP50", "recall@0.25", "AP50-95"]
        w.writerow(["scenario", "model"] + [f"{m}_{k}" for m in mets for k in ("real", "s_test", "gap", "gap_lo", "gap_hi")])
        for scen in ("dataset", "samelabeler"):
            pm = s["rq1"][scen]["per_model"]
            for n in s["rq1_models"]:
                r = pm[n]
                w.writerow([scen, n] + [round(x, 4) for m in mets for x in
                                        (r["real"][m], r["s_test"][m], r["gap"][m], *r["gap_ci95"][m])])


# ---------------------------------------------------------------- fig 3: contrasts real vs S-test
def fig3(s, lang, scen="dataset", name="fig3_rq1_contrasts", sub_key="f3_sub"):
    t = T[lang]
    vc = s["rq1"][scen]["variant_contrasts"]
    metrics = ["AP50", "recall@0.25", "AP50-95"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 6.4 + 0.5 * (len(CONTRASTS) - 5)), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    off = 0.17
    for j, m in enumerate(metrics):
        ax = axes[j]
        style(ax)
        for i, c in enumerate(CONTRASTS):
            d = vc[c][m]
            for k, (key, col, mk, lab) in enumerate([("real", BLUE, "o", t["real"]), ("s_test", ORANGE, "s", t["stest"])]):
                y = -i + (off if k == 0 else -off)
                lo, hi = d[f"{key}_ci95"]
                ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
                ax.plot(d[key], y, mk, ms=8, color=col, mec=SURFACE, mew=1.8, zorder=3,
                        label=lab if (i == 0 and j == 0) else None)
        ax.axvline(0, color=INK2, lw=1, zorder=1)
        ax.set_title(MNAME[m], loc="left", fontsize=11, fontweight="bold", color=INK, pad=4)
        ax.set_xlabel(t["diff"], color=INK2, fontsize=9.5)
        lo = min(min(vc[c][m]["real_ci95"][0], vc[c][m]["s_test_ci95"][0]) for c in CONTRASTS)
        hi = max(max(vc[c][m]["real_ci95"][1], vc[c][m]["s_test_ci95"][1]) for c in CONTRASTS)
        ax.set_xlim(lo - 0.02, hi + 0.02)
        for i in range(len(CONTRASTS) - 1):
            ax.axhline(-i - 0.5, color=GRID, lw=0.8)
    axes[0].set_yticks([-i for i in range(len(CONTRASTS))], [contrast_label(c, lang) for c in CONTRASTS])
    axes[0].tick_params(axis="y", labelsize=10, labelcolor=INK, pad=8)
    axes[0].set_ylim(-len(CONTRASTS) + 0.5, 0.5)
    fig.legend(loc="upper left", bbox_to_anchor=(0.02, 0.86), ncol=2, frameon=False, fontsize=9.5,
               labelcolor=INK, handletextpad=0.4, columnspacing=1.6)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.77, bottom=0.1, wspace=0.08)
    header(fig, t["f3_title"], t[sub_key])
    save(fig, name, lang)


def fig3_csv(s):
    with open(OUT / "fig3_rq1_contrasts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "contrast", "metric", "real", "real_lo", "real_hi", "s_test", "s_test_lo", "s_test_hi",
                    "s_test_minus_real", "smr_lo", "smr_hi"])
        for scen in ("dataset", "samelabeler"):
            for c, mm in s["rq1"][scen]["variant_contrasts"].items():
                for m, d in mm.items():
                    w.writerow([scen, c, m] + [round(x, 4) for x in (d["real"], *d["real_ci95"], d["s_test"], *d["s_test_ci95"],
                                                                    d["s_test_minus_real"], *d["s_test_minus_real_ci95"])])


# ---------------------------------------------------------------- fig 4: pair agreement
def pair_parts(p):
    opp = p["both_sure_opposite"]
    real_only = p["real_sure"] - p["both_sure_same"] - opp
    none = p["n"] - p["both_sure_same"] - p["syn_sure_real_not"] - real_only - opp
    return [p["both_sure_same"], p["syn_sure_real_not"], real_only, opp, none]


def fig4(s, lang):
    t = T[lang]
    cats = [(t["p_both"], BLUE), (t["p_syn"], ORANGE), (t["p_real"], AQUA), (t["p_opp"], VIOLET), (t["p_none"], NEUTRAL)]
    rows = []
    for m in ["AP50", "recall@0.25", "AP50-95"]:
        for scen, sk in (("dataset", "sc_dataset"), ("samelabeler", "sc_same")):
            a = s["rq1"][scen]["aggregate"][m]
            rows.append((f"{MNAME[m]} · {t[sk]}", pair_parts(a["pairs"]), a["pairs"]["n"], a["kendall_tau_b"],
                         a["kendall_tau_b_ci95"]))
    fig, ax = plt.subplots(figsize=(12, 5.6))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    gap = 0.004  # surface gap between segments
    ypos = []
    y = 0.0
    for r, (lab, parts, n, tau, tci) in enumerate(rows):
        if r and r % 2 == 0:
            y -= 0.45
        ypos.append(y)
        left = 0.0
        for (cname, col), v in zip(cats, parts):
            w = v / n
            if w > 0:
                ax.barh(y, max(w - gap, 0.0015), left=left, height=0.72, color=col, edgecolor="none",
                        label=cname if r == 0 else None, zorder=2)
                if w >= 0.05:
                    tc = INK if col in (NEUTRAL, AQUA, ORANGE) else "#ffffff"
                    ax.text(left + w / 2, y, f"{v}", ha="center", va="center", fontsize=8.5, color=tc, zorder=3)
            left += w
        ax.text(1.015, y, f"τ = {tau:.2f}  [{tci[0]:.2f}, {tci[1]:.2f}]\n{t['opp_short']}: {parts[3]}", va="center",
                fontsize=9, color=INK, linespacing=1.4,
                transform=ax.get_yaxis_transform())
        y -= 1.0
    ax.set_yticks(ypos, [r[0] for r in rows])
    ax.tick_params(axis="y", labelsize=9.5, labelcolor=INK, pad=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(y + 0.4, 0.6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(t["share"], color=INK2, fontsize=9.5)
    h, l = ax.get_legend_handles_labels()
    order = [0, 3, 1, 4, 2]  # matplotlib fills legend columns first; this reads row by row in segment order
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", bbox_to_anchor=(0.02, 0.86), ncol=3,
               frameon=False, fontsize=9, labelcolor=INK, handletextpad=0.4, columnspacing=1.4)
    fig.subplots_adjust(left=0.23, right=0.83, top=0.72, bottom=0.11)
    header(fig, t["f4_title"], t["f4_sub"].format(**COUNTS))
    save(fig, "fig4_rq1_pairs", lang)


def fig4_csv(s):
    with open(OUT / "fig4_rq1_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "scenario", "pairs", "both_same", "s_test_only", "real_only", "both_opposite", "neither",
                    "kendall_tau_b", "tau_lo", "tau_hi"])
        for m in ["AP50", "recall@0.25", "AP50-95"]:
            for scen in ("dataset", "samelabeler"):
                a = s["rq1"][scen]["aggregate"][m]
                w.writerow([m, scen, a["pairs"]["n"], *pair_parts(a["pairs"]), round(a["kendall_tau_b"], 4),
                            round(a["kendall_tau_b_ci95"][0], 4), round(a["kendall_tau_b_ci95"][1], 4)])


def main():
    global PAPER
    import sys
    PAPER = "--paper" in sys.argv[1:]
    s = json.loads(SUMMARY.read_text())
    configure(s)
    OUT.mkdir(parents=True, exist_ok=True)
    for lang in (("en",) if PAPER else ("ru", "en")):
        fig1(s, lang)
        fig2(s, lang)
        fig2(s, lang, m="recall@0.25", name="fig2b_rq1_scatter_recall")
        fig3(s, lang)
        fig3(s, lang, scen="samelabeler", name="fig3b_rq1_contrasts_samelabeler", sub_key="f3b_sub")
        fig4(s, lang)
    if PAPER:
        print("\n".join(sorted(p.name for p in (OUT / "paper").iterdir())))
        return
    fig1_csv(s)
    fig2_csv(s)
    fig3_csv(s)
    fig4_csv(s)
    print("\n".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
