"""Per-batch summary of the synthetic data for the article (quality, rejection, memorization, diversity, RQ3).

    .venv/bin/python scripts/batch_table.py        (after rq3_classifier.py)

Batches: s_test, train_low, train_full (prompt v4). Everything is computed on the *accepted* images
(manifest status != rejected) with the same functions as analyze_pilot.py, so the numbers are
consistent across batches (the older viz/<batch>_stats.json files include rejected images or were
computed on part of a batch).
Reference row: real glioma images of subset_v1 train (the pool the train batches were drawn from).

Columns:
  generated / rejected / accepted         manifest rows (rejected_in_blind_review: s_test 0040, accepted with a draft)
  unchanged / corrected / drawn           human box review against the bonus-model draft
  detected@0.25                           bonus detector (trained on real data) found >=1 glioma box
                                          at conf 0.25 (non-empty labels_draft/<name>.txt)
  multi_lesion                            share of images with >=2 boxes (s_test: blind boxes)
  brain_mean / contrast / sharpness / head_fraction   mean +- sd (analyze_pilot.intensity)
  dhash_to_ref_median, ssim_to_ref_median  similarity to the image's own reference
  dhash_nearest_train_min, near_dups      nearest real train image; near-duplicate threshold 28
  rq3_auc_handcrafted / rq3_auc_detector   per-batch AUC of the real-vs-synthetic classifier
Output: viz/batch_table.json, viz/batch_table.csv
"""
import csv
import json
from pathlib import Path

import numpy as np

from analyze_pilot import NEAR_DUP, ROOT, SUBSET, dhash_bits, intensity, ssim

BATCHES = ("s_test", "train_low", "train_full")


def ms(v):
    v = np.asarray(v, dtype=float)
    return round(float(v.mean()), 2), round(float(v.std()), 2)


def n_boxes(p):
    return sum(1 for l in p.read_text().splitlines() if l.strip()) if p.exists() else None


def ref_path(name):
    for split in ("train", "test", "val"):
        p = SUBSET / "images" / split / name
        if p.exists():
            return p
    raise FileNotFoundError(name)


def main():
    manifest = list(csv.DictReader(open(ROOT / "generated" / "manifest.csv")))
    rq3 = json.loads((ROOT / "viz" / "rq3_classifier.json").read_text())
    train = sorted((SUBSET / "images" / "train").glob("*.jpg"))
    train_bits = np.stack([dhash_bits(p) for p in train])
    rows = []

    real = sorted((SUBSET / "images" / "train").glob("gl_*.jpg"))
    I = np.array([intensity(p) for p in real])
    real_multi = []
    for split in ("train", "val", "test"):
        for p in (SUBSET / "images" / split).glob("gl_*.jpg"):
            n = n_boxes(SUBSET / "labels" / split / (p.stem + ".txt"))
            if n:
                real_multi.append(n >= 2)
    rb = [dhash_bits(p) for p in real]
    rng = np.random.default_rng(0)
    pairs = [int((rb[i] != rb[j]).sum()) for i, j in (rng.choice(len(rb), 2, replace=False) for _ in range(5000))]
    rows.append({"batch": "real glioma (subset_v1 train)", "accepted": len(real),
                 "multi_lesion": round(float(np.mean(real_multi)), 3),
                 **{k: ms(I[:, i]) for i, k in enumerate(["brain_mean", "contrast", "sharpness", "head_fraction"])},
                 "dhash_between_real_median": float(np.median(pairs))})

    for b in BATCHES:
        allr = [r for r in manifest if r["purpose"] == b]
        acc = [r for r in allr if r["status"] != "rejected"]
        box = {k: sum(r["box_checked"] == k for r in acc) for k in ("unchanged", "corrected", "drawn", "blind")}
        lab_dir = ROOT / "generated" / b / ("labels_blind" if b == "s_test" else "labels")
        draft_dir = ROOT / "generated" / b / "labels_draft"
        det, multi, sim = [], [], []
        for r in acc:
            stem = Path(r["file"]).stem
            nd = n_boxes(draft_dir / f"{stem}.txt")
            if nd is not None:
                det.append(nd > 0)
            nb = n_boxes(lab_dir / f"{stem}.txt")
            if nb:
                multi.append(nb >= 2)
            f = ROOT / r["file"]
            bits = dhash_bits(f)
            ref = ref_path(r["refs"].split(";")[0])
            sim.append(((dhash_bits(ref) != bits).sum(), ssim(f, ref), (train_bits != bits).sum(1).min()))
        I = np.array([intensity(ROOT / r["file"]) for r in acc])
        sim = np.array(sim, dtype=float)
        rows.append({
            "batch": b, "generated": len(allr), "rejected": len(allr) - len(acc), "rejected_in_blind_review": sum("(blind)" in r["rejection_reason"] for r in allr),
            "accepted": len(acc), "unique_refs": len({r["refs"] for r in acc}),
            **box,
            "detected@0.25": round(float(np.mean(det)), 3) if det else None, "n_with_draft": len(det),
            "multi_lesion": round(float(np.mean(multi)), 3), "n_labeled": len(multi),
            **{k: ms(I[:, i]) for i, k in enumerate(["brain_mean", "contrast", "sharpness", "head_fraction"])},
            "dhash_to_ref_median": float(np.median(sim[:, 0])), "ssim_to_ref_median": round(float(np.median(sim[:, 1])), 3),
            "dhash_nearest_train_min": int(sim[:, 2].min()), "near_dups": int((sim[:, 2] <= NEAR_DUP).sum()),
            "rq3_auc_handcrafted": round(rq3["handcrafted"]["per_batch_auc"][b], 3),
            "rq3_auc_detector": round(rq3["detector"]["per_batch_auc"][b], 3),
        })

    out = ROOT / "viz" / "batch_table.json"
    out.write_text(json.dumps({"rows": rows, "rq3_overall": {
        "handcrafted": {k: rq3["handcrafted"][k] for k in ("auc", "ci95", "n_real", "n_synthetic")},
        "detector": {k: rq3["detector"][k] for k in ("auc", "ci95", "n_real", "n_synthetic")}}}, indent=1))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(ROOT / "viz" / "batch_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v[0]} ± {v[1]}" if isinstance(v, tuple) else v) for k, v in r.items()})
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
