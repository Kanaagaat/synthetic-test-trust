"""Example figure for the paper: real test gliomas (top) and their synthetic S-test twins (bottom).

    .venv/bin/python scripts/fig_examples.py [--ids 0000 0002 ...]

Top row: real test image with its dataset boxes. Bottom row: the twin generated from it (prompt v4,
reference-guided) with its blind boxes (labels_blind, the boxes used for evaluation).
Output: viz/article/paper/fig0_examples_en.{pdf,png}
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUBSET = (ROOT / "data" / "subset_v1" if (ROOT / "data" / "subset_v1").exists() else ROOT.parent / "bonus_task" / "data" / "subset_v1")
OUT = ROOT / "viz" / "article" / "paper"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def boxes(p):
    if not p.exists():
        return []
    return [list(map(float, l.split()[1:5])) for l in p.read_text().splitlines() if l.strip() and l.split()[0] == "0"]


def draw(ax, img_path, bxs, color):
    im = Image.open(img_path).convert("L").resize((512, 512), Image.LANCZOS)
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    for x, y, w, h in bxs:
        ax.add_patch(Rectangle(((x - w / 2) * 512, (y - h / 2) * 512), w * 512, h * 512,
                               fill=False, ec=color, lw=1.6))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", default=["0000", "0005", "0010", "0020", "0033"])
    a = ap.parse_args()
    man = {Path(r["file"]).stem: r for r in csv.DictReader(open(ROOT / "generated" / "manifest.csv"))
           if r["purpose"] == "s_test" and r["status"] != "rejected"}
    n = len(a.ids)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.45))
    for j, i in enumerate(a.ids):
        stem = f"syn_gl_st_{i}"
        r = man[stem]
        ref = r["refs"].split(";")[0]
        real = SUBSET / "images" / "test" / ref
        draw(axes[0, j], real, boxes(SUBSET / "labels" / "test" / (Path(ref).stem + ".txt")), BLUE)
        draw(axes[1, j], ROOT / r["file"], boxes(ROOT / "generated" / "s_test" / "labels_blind" / f"{stem}.txt"), ORANGE)
    axes[0, 0].set_ylabel("real test", fontsize=10)
    axes[1, 0].set_ylabel("synthetic twin", fontsize=10)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.995, bottom=0.005, wspace=0.03, hspace=0.03)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig0_examples_en.{ext}", dpi=200, facecolor="white")
    print(OUT / "fig0_examples_en.pdf")


if __name__ == "__main__":
    main()
