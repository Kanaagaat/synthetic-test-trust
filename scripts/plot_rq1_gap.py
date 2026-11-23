"""Forest plot of the RQ1 pilot: S-test minus real test, bonus model, glioma, 4 annotation scenarios.

    .venv/bin/python scripts/plot_rq1_gap.py            # writes viz/rq1_gap_ru.png and viz/rq1_gap_en.png

Colors: reference categorical slots 1-4 (validated: adjacent CVD dE >= 9.1, normal-vision >= 22.9 on
#fcfcfb). Slots 3-4 are below 3:1 contrast on the surface, so every series also has its own marker
shape, a legend, and the numbers are printed in a table column on the right (text ink, not series color).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RQ1 = ROOT / "viz" / "rq1"

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "D", "^"]

SCENARIOS = [  # (result file, ru label, en label)
    ("bonus_exp_epochs100", "Рамки датасета, все пары", "Dataset boxes, all pairs"),
    ("bonus_exp_epochs100__samelabeler_subset_datasetboxes", "Рамки датасета, только где человек видит опухоль",
     "Dataset boxes, only where the annotator sees a tumor"),
    ("bonus_exp_epochs100__samelabeler", "Один разметчик вслепую (реальные и двойники)",
     "One blind annotator (real and twins)"),
    ("bonus_exp_epochs100__assisted__common", "Двойники размечены по черновикам модели",
     "Twins labeled from the model's draft boxes"),
]
METRICS = [("AP50", "AP50"), ("recall@0.25", "Recall\n(conf ≥ 0.25)"), ("AP50-95", "AP50-95")]

TEXT = {
    "ru": {"title": "Синтетический тест завышает качество, а AP50-95 зависит от того, кто ставил рамки",
           "sub": "Бонус-модель YOLO11n, класс «глиома». Разрыв = метрика на S-test (двойники) − на реальном тесте. "
                  "Линии — 95% ДИ, 2000 bootstrap-ресэмплов групп.",
           "x": "Разрыв: S-test − реальный тест", "zero": "нет разницы", "col": "разрыв [95% ДИ] · пар",
           "better": "S-test завышает →", "worse": "← S-test занижает"},
    "en": {"title": "A synthetic test overstates detection; AP50-95 depends on who drew the boxes",
           "sub": "Bonus YOLO11n model, glioma class. Gap = metric on the S-test (synthetic twins) − on the real test. "
                  "Lines: 95% CI, 2000 group-bootstrap resamples.",
           "x": "Gap: S-test − real test", "zero": "no difference", "col": "gap [95% CI] · pairs",
           "better": "S-test overstates →", "worse": "← S-test understates"},
}


def fmt(v):
    v = 0.0 if abs(v) < 0.005 else v  # no "-0.00"
    return f"{v:+.2f}".replace("-", "−")


def load():
    out = []
    for f, ru, en in SCENARIOS:
        d = json.loads((RQ1 / f"{f}.json").read_text())
        out.append({"ru": ru, "en": en, "pairs": d["n_glioma_images"],
                    "gap": d["gap"], "ci": d["ci95"]["gap"]})
    return out


def plot(lang, data):
    t = TEXT[lang]
    fig, ax = plt.subplots(figsize=(11, 6.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    step, gap_between = 0.6, 1.0
    ytick, ylab = [], []
    y0 = 0.0
    rows = []
    for mi, (mk, mname) in enumerate(METRICS):
        ys = [y0 - i * step for i in range(len(data))]
        for si, s in enumerate(data):
            lo, hi = s["ci"][mk]
            g = s["gap"][mk]
            ax.plot([lo, hi], [ys[si]] * 2, color=SERIES[si], lw=2, solid_capstyle="round", zorder=2)
            ax.plot(g, ys[si], MARKERS[si], ms=10 if MARKERS[si] == "^" else 8, color=SERIES[si],
                    mec=SURFACE, mew=2, zorder=3,
                    label=s[lang] if mi == 0 else None)
            rows.append((ys[si], f"{fmt(g)} [{fmt(lo)}, {fmt(hi)}] · {s['pairs']}"))
        ytick.append(sum(ys) / len(ys))
        ylab.append(mname)
        y0 = ys[-1] - step - gap_between

    ax.axvline(0, color=INK2, lw=1, zorder=1)
    xmin, xmax = -0.25, 0.47
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(y0 + gap_between - 0.2, 0.55)
    ax.set_yticks(ytick, ylab, fontsize=12, color=INK, fontweight="bold", linespacing=1.3)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", colors=INK2, labelsize=10, length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel(t["x"], color=INK2, fontsize=11, labelpad=8)
    ax.text(0.005, 0.01, t["zero"], color=INK2, fontsize=9, va="bottom", ha="left", transform=ax.get_xaxis_transform())
    ax.text(xmax, 1.0, t["better"], color=MUTED, fontsize=9, ha="right", va="bottom", transform=ax.get_xaxis_transform())
    ax.text(xmin, 1.0, t["worse"], color=MUTED, fontsize=9, ha="left", va="bottom", transform=ax.get_xaxis_transform())

    # numbers column (table view), text ink
    tx = 1.02
    ax.text(tx, 1.0, t["col"], transform=ax.transAxes, color=INK2, fontsize=9, va="bottom", ha="left")
    for y, s in rows:
        ax.text(tx, y, s, transform=ax.get_yaxis_transform(), color=INK, fontsize=9, va="center", ha="left",
                family="DejaVu Sans Mono")

    fig.suptitle(t["title"], x=0.02, y=0.985, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.02, 0.935, t["sub"], ha="left", fontsize=9.5, color=INK2, wrap=True)
    fig.legend(loc="upper left", bbox_to_anchor=(0.02, 0.91), ncol=2, frameon=False, fontsize=9.5,
               labelcolor=INK, handletextpad=0.4, columnspacing=1.6)
    fig.subplots_adjust(left=0.15, right=0.70, top=0.80, bottom=0.10)
    out = ROOT / "viz" / f"rq1_gap_{lang}.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main():
    data = load()
    for lang in ("ru", "en"):
        print(plot(lang, data))


if __name__ == "__main__":
    main()
