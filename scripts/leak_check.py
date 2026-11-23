"""Leakage check: are synthetic TRAIN images suspiciously close to real val/test images?

    .venv/bin/python scripts/leak_check.py

Our own leakage path (the reference image) is closed by design: train synthetic is generated only from
train references. What we cannot control is the generator's pretraining: the dataset is public, so the
generator may have seen val/test images. If it memorized them, synthetic train images would land close
to some val/test image. We measure, for every accepted image of train_low and train_full:
  nearest val/test image by dHash (256 bits, same hash as the subset split; near-duplicate <= 28)
  max SSIM (analyze_pilot.ssim) over the 10 dHash-nearest val/test images
and compare with the same numbers for
  baseline A  real train gliomas -> nearest val/test (different split by construction: what "unrelated
              but same dataset" looks like)
  baseline B  the same synthetic image -> nearest train image other than its own reference
  own ref     synthetic -> its own reference (the image it was made from)
Flagged for inspection: within the near-duplicate threshold of a val/test image, or SSIM to a val/test
image above the maximum of baseline A. ("Closer to some val/test image than to its own reference" is
reported but is NOT a leak signal: a minimum over ~400 images is almost always below the distance to one
given image.) Baseline B is also given with a pool-size-matched random train subset (same size as
val+test), since the train pool is 2.3x larger. s_test twins are reported for information only (they are
made from test images on purpose and never enter training).
Output: viz/leak_check.json, viz/leak_check_closest.jpg (closest synthetic/val-test pairs)
"""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from analyze_pilot import NEAR_DUP, ROOT, SUBSET, dhash_bits, gray, ssim

K = 10


def main():
    manifest = [r for r in csv.DictReader(open(ROOT / "generated" / "manifest.csv"))
                if r["purpose"] in ("train_low", "train_full", "s_test") and r["status"] != "rejected"]
    pools = {s: sorted((SUBSET / "images" / s).glob("*.jpg")) for s in ("train", "val", "test")}
    bits = {s: np.stack([dhash_bits(p) for p in ps]) for s, ps in pools.items()}
    vt_paths = pools["val"] + pools["test"]
    vt_bits = np.concatenate([bits["val"], bits["test"]])
    train_name = [p.name for p in pools["train"]]

    def nearest_vt(path, b):
        d = (vt_bits != b).sum(1)
        top = np.argsort(d)[:K]
        ss = [ssim(path, vt_paths[i]) for i in top]
        j = int(np.argmax(ss))
        return {"dhash_min": int(d[top[0]]), "dhash_nearest": vt_paths[top[0]].name,
                "ssim_max": round(float(ss[j]), 3), "ssim_nearest": vt_paths[top[j]].name,
                "nearest_split": "val" if top[0] < len(pools["val"]) else "test"}

    rows = []
    rng = np.random.default_rng(0)
    for r in manifest:
        f = ROOT / r["file"]
        b = dhash_bits(f)
        ref = r["refs"].split(";")[0]
        refp = next(SUBSET / "images" / s / ref for s in ("train", "test", "val") if (SUBSET / "images" / s / ref).exists())
        dtr = (bits["train"] != b).sum(1)
        mask = np.array([n != ref for n in train_name])
        top_tr = np.argsort(np.where(mask, dtr, 999))[:K]
        sub = rng.choice(np.flatnonzero(mask), len(vt_paths), replace=False)
        rows.append({"file": r["file"], "batch": r["purpose"], "ref": ref,
                     "dhash_nearest_train_other_matched": int(dtr[sub].min()),
                     "dhash_to_ref": int((dhash_bits(refp) != b).sum()), "ssim_to_ref": round(ssim(f, refp), 3),
                     "dhash_nearest_train_other": int(dtr[top_tr[0]]),
                     "ssim_max_train_other": round(max(ssim(f, pools["train"][i]) for i in top_tr), 3),
                     **{f"vt_{k}": v for k, v in nearest_vt(f, b).items()}})

    base = []
    for p, b in zip(pools["train"], bits["train"]):
        if p.name.startswith("gl_"):
            base.append({"file": p.name, **{f"vt_{k}": v for k, v in nearest_vt(p, b).items()}})

    def summ(xs, key):
        v = np.array([x[key] for x in xs], float)
        return {"min": float(v.min()), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)),
                "max": float(v.max())} if len(v) else None

    out = {"near_dup_threshold": NEAR_DUP, "k_ssim_candidates": K,
           "baseline_real_train_glioma_to_valtest": {"n": len(base), "dhash": summ(base, "vt_dhash_min"),
                                                     "ssim_max": summ(base, "vt_ssim_max")},
           "batches": {}}
    for bt in ("train_low", "train_full", "s_test"):
        xs = [x for x in rows if x["batch"] == bt]
        bmax = out["baseline_real_train_glioma_to_valtest"]["ssim_max"]["max"]
        flag = [x for x in xs if x["vt_dhash_min"] <= NEAR_DUP or x["vt_ssim_max"] > bmax]
        out["batches"][bt] = {
            "n": len(xs),
            "valtest_dhash": summ(xs, "vt_dhash_min"), "valtest_ssim_max": summ(xs, "vt_ssim_max"),
            "train_other_dhash": summ(xs, "dhash_nearest_train_other"),
            "train_other_dhash_pool_matched": summ(xs, "dhash_nearest_train_other_matched"), "train_other_ssim_max": summ(xs, "ssim_max_train_other"),
            "own_ref_dhash": summ(xs, "dhash_to_ref"), "own_ref_ssim": summ(xs, "ssim_to_ref"),
            "n_valtest_near_dup": sum(x["vt_dhash_min"] <= NEAR_DUP for x in xs),
            "n_closer_to_valtest_than_ref_dhash": sum(x["vt_dhash_min"] < x["dhash_to_ref"] for x in xs),
            "n_closer_to_valtest_than_ref_ssim": sum(x["vt_ssim_max"] >= x["ssim_to_ref"] for x in xs),
            "share_valtest_ssim_above_baseline_max": float(np.mean([x["vt_ssim_max"] > out["baseline_real_train_glioma_to_valtest"]["ssim_max"]["max"] for x in xs])),
            "flagged": sorted(flag, key=lambda x: -x["vt_ssim_max"]),
        }
    out["per_image"] = rows
    (ROOT / "viz" / "leak_check.json").write_text(json.dumps(out, indent=1))

    # sheet: 12 train synthetic images with the highest SSIM to any val/test image
    top = sorted([x for x in rows if x["batch"] != "s_test"], key=lambda x: -x["vt_ssim_max"])[:12]
    T = 256
    sheet = Image.new("L", (3 * T, len(top) * (T + 18)), 255)
    d = ImageDraw.Draw(sheet)
    for i, x in enumerate(top):
        y = i * (T + 18)
        vt = next(p for p in vt_paths if p.name == x["vt_ssim_nearest"])
        refp = next(SUBSET / "images" / s / x["ref"] for s in ("train", "test", "val") if (SUBSET / "images" / s / x["ref"]).exists())
        for j, (p, lab) in enumerate(((ROOT / x["file"], f"{Path(x['file']).stem}"), (refp, f"own ref ssim {x['ssim_to_ref']}"),
                                      (vt, f"val/test ssim {x['vt_ssim_max']}"))):
            sheet.paste(gray(p, T), (j * T, y + 18))
            d.text((j * T + 4, y + 3), lab, fill=0)
    sheet.save(ROOT / "viz" / "leak_check_closest.jpg", quality=88)

    print(json.dumps({k: v for k, v in out.items() if k not in ("per_image",)} | {"batches": {
        b: {k: v for k, v in d.items() if k != "flagged"} | {"n_flagged": len(d["flagged"])} for b, d in out["batches"].items()}},
        indent=1))


if __name__ == "__main__":
    main()
