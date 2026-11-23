"""Build the RQ2 training sets as small YOLO datasets (symlinked images, copied labels).

    python3 scripts/build_train_sets.py            # builds every variant whose inputs exist
    python3 scripts/build_train_sets.py A_low C_low

Every variant uses the REAL val and test of subset_v1 (absolute paths in data.yaml), so model
selection and the main test never see synthetic images. Only the train split changes:

    A_full  all 945 real train images
    A_low   real train, but only the low_keep gliomas (61 of 245); other classes and backgrounds unchanged
    B_low   A_low + extra copies of the low_keep gliomas, as many as C_low has synthetic images
            (YOLO's online augmentation makes each copy look different) -> "no generator" baseline
    C_low   A_low + the labeled synthetic gliomas generated from low_keep (generated/train_low)
    C_full  A_full + labeled synthetic gliomas generated from R_train (generated/train_full)   [needs that batch]
    D       real train without any real glioma + those R_train synthetic gliomas            [needs that batch]

B_low and C_low get the same number of extra glioma images, so they differ only in where the extras
come from (copies of real images vs new synthetic ones).
"""
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBSET = (ROOT / "data" / "subset_v1" if (ROOT / "data" / "subset_v1").exists() else ROOT.parent / "bonus_task" / "data" / "subset_v1")
NAMES = {0: "Glioma", 1: "Meningioma", 2: "Pituitary"}


def real_train():
    rows = [r for r in csv.DictReader(open(SUBSET / "split_manifest.csv")) if r["split"] == "train"]
    return [(SUBSET / "images" / "train" / r["file"], SUBSET / "labels" / "train" / (Path(r["file"]).stem + ".txt"),
             r["classes_new"] == "0") for r in rows]


def synthetic(purpose):
    out = []
    for r in csv.DictReader(open(ROOT / "generated" / "manifest.csv")):
        if r["purpose"] != purpose or r["status"] != "labeled":
            continue
        img = ROOT / r["file"]
        lab = img.parent / "labels" / (img.stem + ".txt")
        if lab.exists():
            out.append((img, lab, True))
    return out


def write(name, items, note):
    d = ROOT / "datasets" / f"rq2_{name}"
    if d.exists():
        shutil.rmtree(d)
    (d / "images" / "train").mkdir(parents=True)
    (d / "labels" / "train").mkdir(parents=True)
    seen = set()
    for img, lab, _ in items:
        stem = img.stem
        k = 1
        while stem in seen:  # duplicated real images get a suffix
            k += 1
            stem = f"{img.stem}_copy{k}"
        seen.add(stem)
        (d / "images" / "train" / f"{stem}{img.suffix}").symlink_to(img)
        (d / "labels" / "train" / f"{stem}.txt").write_text(lab.read_text() if lab.exists() else "")
    (d / "data.yaml").write_text(
        f"# RQ2 variant {name}: {note}\npath: {d}\ntrain: images/train\n"
        f"val: {SUBSET / 'images' / 'val'}\ntest: {SUBSET / 'images' / 'test'}\nnames:\n"
        + "".join(f"  {k}: {v}\n" for k, v in NAMES.items()))
    n_gl = sum(g for *_, g in items)
    print(f"{name:7s} train {len(items):4d} images, {n_gl:3d} with glioma  -> {d / 'data.yaml'}")
    return {"name": name, "train_images": len(items), "glioma_images": n_gl, "note": note}


def main():
    want = set(sys.argv[1:])
    pools = json.loads((ROOT / "refs" / "pools.json").read_text())
    keep = set(pools["low_keep"])
    train = real_train()
    non_glioma = [x for x in train if not x[2]]
    low_real = non_glioma + [x for x in train if x[2] and x[0].name in keep]
    syn_low, syn_full = synthetic("train_low"), synthetic("train_full")
    built = []

    def go(name):
        return not want or name in want

    if go("A_full"):
        built.append(write("A_full", train, "all real train images"))
    if go("A_low"):
        built.append(write("A_low", low_real, "real train, gliomas reduced to low_keep (~25%)"))
    if syn_low:
        kept = [x for x in train if x[2] and x[0].name in keep]
        copies = [kept[i % len(kept)] for i in range(len(syn_low))]
        if go("B_low"):
            built.append(write("B_low", low_real + copies, f"A_low + {len(copies)} extra copies of low_keep gliomas"))
        if go("C_low"):
            built.append(write("C_low", low_real + syn_low, f"A_low + {len(syn_low)} labeled synthetic gliomas"))
    else:
        print("B_low / C_low: no labeled synthetic images in generated/train_low yet")
    if syn_full:
        if go("C_full"):
            built.append(write("C_full", train + syn_full, f"A_full + {len(syn_full)} labeled synthetic gliomas"))
        if go("D"):
            built.append(write("D", non_glioma + syn_full, f"real train without gliomas + {len(syn_full)} synthetic gliomas"))
    else:
        print("C_full / D: no labeled synthetic images in generated/train_full yet")
    (ROOT / "datasets" / "rq2_variants.json").write_text(json.dumps(built, indent=1))


if __name__ == "__main__":
    main()
