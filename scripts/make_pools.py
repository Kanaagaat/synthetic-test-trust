"""Reference pools for generation (v2 of the pool design).

R_train      all train glioma images -> references for synthetic *train* images (C_full, D)
low_keep     ~25% of train glioma images, sampled by leakage group -> the real gliomas the
             low regime keeps; synthetic for C_low is generated only from these
S_test_refs  the real *test* glioma images -> each gets one synthetic "twin" for the S-test

Why test images as S-test references: the S-test then has the same tumor types, sizes and
planes as the real test, image by image, so the real-vs-synthetic comparison is paired and
not driven by our prompt mix. The twins are only ever evaluated on, never trained on or used
for model selection, and no model sees the test images themselves.
"""
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBSET = (ROOT / "data" / "subset_v1" if (ROOT / "data" / "subset_v1").exists() else ROOT.parent / "bonus_task" / "data" / "subset_v1")
SEED, LOW_SHARE = 2026, 0.25

rows = list(csv.DictReader(open(SUBSET / "split_manifest.csv")))
gl = lambda split: [r for r in rows if r["split"] == split and r["classes_new"] == "0"]  # noqa: E731

train = gl("train")
groups = sorted({r["group"] for r in train}, key=int)
rng = random.Random(SEED)
rng.shuffle(groups)
keep, n_target = set(), round(LOW_SHARE * len(train))
for g in groups:  # add whole groups until ~25% of images
    if sum(r["group"] in keep for r in train) >= n_target:
        break
    keep.add(g)

pools = {
    "seed": SEED,
    "note": "glioma-only images (classes_new == '0') of bonus_task/data/subset_v1",
    "R_train": sorted(r["file"] for r in train),
    "low_keep": sorted(r["file"] for r in train if r["group"] in keep),
    "S_test_refs": sorted(r["file"] for r in gl("test")),
}
pools["counts"] = {k: len(v) for k, v in pools.items() if isinstance(v, list)}
pools["counts"]["low_keep_groups"] = len(keep)
(ROOT / "refs" / "pools.json").write_text(json.dumps(pools, indent=1))
print(pools["counts"])
