"""Pilot review helpers: contact sheet, diversity stats and memorization check.

    python scripts/analyze_pilot.py --purpose pilot_v2 --refs-json refs/pilot_v2/refs.json

Outputs:
    viz/<purpose>_sheet.jpg      each row: reference | its synthetic variants
    viz/<purpose>_stats.json     intensity diversity vs real gliomas, similarity numbers
Similarity:
    dHash distance (256 bits, same hash as the bonus-task split) and SSIM (global
    11x11 box-window SSIM on 256x256 grayscale) to (a) the image's own reference and
    (b) the nearest real train image. Train near-duplicates in subset_v1 were
    grouped at dHash <= 28, so a synthetic image that close to a train image is
    treated as a likely copy.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SUBSET = (ROOT / "data" / "subset_v1" if (ROOT / "data" / "subset_v1").exists() else ROOT.parent / "bonus_task" / "data" / "subset_v1")
NEAR_DUP = 28


def gray(path, size=None):
    im = Image.open(path).convert("L")
    return im.resize((size, size), Image.LANCZOS) if size else im


def dhash_bits(path, size=16):
    a = np.asarray(Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel()


def box_mean(a, k=11):
    c = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / (k * k)


def ssim(p, q, size=256):
    x = np.asarray(gray(p, size), dtype=np.float64)
    y = np.asarray(gray(q, size), dtype=np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = box_mean(x), box_mean(y)
    vx = box_mean(x * x) - mx ** 2
    vy = box_mean(y * y) - my ** 2
    cov = box_mean(x * y) - mx * my
    m = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return float(m.mean())


def intensity(path):
    im = gray(path, 512)
    a = np.asarray(im, dtype=float)
    fg = a > 20
    edges = np.asarray(im.filter(ImageFilter.FIND_EDGES), dtype=float)
    return a[fg].mean(), a[fg].std(), edges.var(), fg.mean()


def summarize(vals):
    s = np.array(vals)
    names = ["brain_mean", "contrast_std", "edge_var", "head_fraction"]
    return {n: {"mean": round(float(s[:, i].mean()), 2), "sd": round(float(s[:, i].std()), 2)}
            for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--purpose", default="pilot_v2")
    ap.add_argument("--refs-json", default=None, help="pilot v2 refs list; omit for pool-mode purposes")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(ROOT / "generated" / "manifest.csv")) if r["purpose"] == args.purpose]
    if args.refs_json:
        ref_dir = (ROOT / args.refs_json).parent
        refs = json.loads((ROOT / args.refs_json).read_text())
    else:  # v3 pool mode: references live in the subset itself
        seen = {}
        for r in rows:
            seen.setdefault(r["refs"], {"ref": r["refs"], "plane": "", "size": r["size"], "ref_pattern": ""})
        refs = list(seen.values())
        ref_dir = None

    def ref_path(name):
        if ref_dir is not None and (ref_dir / name).exists():
            return ref_dir / name
        for split in ("train", "test", "val"):
            p = SUBSET / "images" / split / name
            if p.exists():
                return p
        raise FileNotFoundError(name)

    # contact sheet: reference + variants per row
    T, per = 256, max(1, max(sum(r["refs"] == x["ref"] for r in rows) for x in refs))
    sheet = Image.new("L", ((per + 1) * T, len(refs) * (T + 18)), 255)
    d = ImageDraw.Draw(sheet)
    for i, x in enumerate(refs):
        y = i * (T + 18)
        sheet.paste(gray(ref_path(x["ref"]), T), (0, y + 18))
        d.text((4, y + 3), f"REF {x['plane']} {x['size']} {x['ref_pattern']}", fill=0)
        for j, r in enumerate(r for r in rows if r["refs"] == x["ref"]):
            sheet.paste(gray(ROOT / r["file"], T), ((j + 1) * T, y + 18))
            d.text(((j + 1) * T + 4, y + 3), f"{Path(r['file']).stem[-4:]} {r['acquisition'][:30]}", fill=0)
    # JPEG height is capped at 65500 px: split long batches into pages (_sheet_p1.jpg, ...)
    rows_per_page = 65000 // (T + 18)
    if len(refs) <= rows_per_page:
        sheet_path = ROOT / "viz" / f"{args.purpose}_sheet.jpg"
        sheet.save(sheet_path, quality=90)
    else:
        for p, i0 in enumerate(range(0, len(refs), rows_per_page), 1):
            n = min(rows_per_page, len(refs) - i0)
            sheet_path = ROOT / "viz" / f"{args.purpose}_sheet_p{p}.jpg"
            sheet.crop((0, i0 * (T + 18), sheet.width, (i0 + n) * (T + 18))).save(sheet_path, quality=90)

    # similarity
    train = sorted((SUBSET / "images" / "train").glob("*.jpg"))
    train_bits = np.stack([dhash_bits(p) for p in train])
    per_image = []
    for r in rows:
        f = ROOT / r["file"]
        b = dhash_bits(f)
        dist = (train_bits != b).sum(1)
        k = int(dist.argmin())
        ref = ref_path(r["refs"].split(";")[0])
        per_image.append({
            "file": r["file"], "ref": r["refs"],
            "dhash_to_ref": int((dhash_bits(ref) != b).sum()),
            "ssim_to_ref": round(ssim(f, ref), 3),
            "nearest_train": train[k].name, "dhash_to_nearest_train": int(dist[k]),
            "ssim_to_nearest_train": round(ssim(f, train[k]), 3),
        })

    # baseline: how similar are two *different* real train gliomas (same plane unknown) -> dHash between refs
    ref_paths = [ref_path(x["ref"]) for x in refs]
    ref_pairs = [int((dhash_bits(a) != dhash_bits(b)).sum()) for i, a in enumerate(ref_paths) for b in ref_paths[i + 1:]]

    real_gl = sorted((SUBSET / "images" / "train").glob("gl_*.jpg"))
    stats = {
        "purpose": args.purpose, "n": len(rows),
        "real_glioma_train": summarize([intensity(p) for p in real_gl]),
        "synthetic": summarize([intensity(ROOT / r["file"]) for r in rows]),
        "similarity": {
            "dhash_to_ref_median": float(np.median([p["dhash_to_ref"] for p in per_image])),
            "dhash_to_nearest_train_min": int(min(p["dhash_to_nearest_train"] for p in per_image)),
            "n_within_near_dup_threshold": sum(p["dhash_to_nearest_train"] <= NEAR_DUP for p in per_image),
            "ssim_to_ref_median": float(np.median([p["ssim_to_ref"] for p in per_image])),
            "dhash_between_different_refs_median": float(np.median(ref_pairs)),
        },
        "per_image": per_image,
    }
    out = ROOT / "viz" / f"{args.purpose}_stats.json"
    out.write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "per_image"}, indent=1))
    print("sheet:", sheet_path)


if __name__ == "__main__":
    main()
