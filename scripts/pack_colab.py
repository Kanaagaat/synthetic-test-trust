"""Pack everything the Colab notebook needs into colab/rq2_data.zip.

    python3 scripts/pack_colab.py

Layout inside the zip (one copy of each image; variants are train-list files):
    rq2_data/real/{images,labels}/{train,val,test}/   subset_v1 as is
    rq2_data/syn/{images,labels}/<purpose>/           labeled synthetic gliomas (train_low, train_full)
    rq2_data/dup/{images,labels}/B_low/               physical copies of low_keep gliomas for B_low
    rq2_data/<variant>.txt                            train list, paths "./real/images/train/..." etc.
    rq2_data/variants.json                            counts + notes per variant

Ultralytics resolves "./" in a list file relative to the list's folder and finds each label by
swapping /images/ for /labels/, so this layout trains exactly the sets build_train_sets.py describes.
val and test are always the real ones.
"""
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBSET = (ROOT / "data" / "subset_v1" if (ROOT / "data" / "subset_v1").exists() else ROOT.parent / "bonus_task" / "data" / "subset_v1")
OUT = ROOT / "colab"


def synthetic(purpose):
    out = []
    for r in csv.DictReader(open(ROOT / "generated" / "manifest.csv")):
        if r["purpose"] == purpose and r["status"] == "labeled":
            img = ROOT / r["file"]
            lab = img.parent / "labels" / (img.stem + ".txt")
            if lab.exists():
                out.append((img, lab))
    return out


def main():
    stage = OUT / "rq2_data"
    if stage.exists():
        shutil.rmtree(stage)
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            shutil.copytree(SUBSET / kind / split, stage / "real" / kind / split)

    pools = json.loads((ROOT / "refs" / "pools.json").read_text())
    keep = set(pools["low_keep"])
    rows = [r for r in csv.DictReader(open(SUBSET / "split_manifest.csv")) if r["split"] == "train"]
    real = [f"./real/images/train/{r['file']}" for r in rows]
    non_glioma = [f"./real/images/train/{r['file']}" for r in rows if r["classes_new"] != "0"]
    kept = [r["file"] for r in rows if r["classes_new"] == "0" and r["file"] in keep]
    low = non_glioma + [f"./real/images/train/{f}" for f in kept]

    variants = {"A_full": (real, "all real train images"),
                "A_low": (low, "real train, gliomas reduced to low_keep (~25%)")}

    def add_syn(purpose):
        items = synthetic(purpose)
        for kind in ("images", "labels"):
            (stage / "syn" / kind / purpose).mkdir(parents=True, exist_ok=True)
        for img, lab in items:
            shutil.copy2(img, stage / "syn" / "images" / purpose / img.name)
            shutil.copy2(lab, stage / "syn" / "labels" / purpose / lab.name)
        return [f"./syn/images/{purpose}/{img.name}" for img, _ in items]

    syn_low = add_syn("train_low")
    if syn_low:
        dups = []
        for kind in ("images", "labels"):
            (stage / "dup" / kind / "B_low").mkdir(parents=True, exist_ok=True)
        for i in range(len(syn_low)):
            f = Path(kept[i % len(kept)])
            name = f"{f.stem}_copy{i // len(kept) + 2}"
            shutil.copy2(SUBSET / "images" / "train" / f.name, stage / "dup" / "images" / "B_low" / f"{name}{f.suffix}")
            shutil.copy2(SUBSET / "labels" / "train" / f"{f.stem}.txt", stage / "dup" / "labels" / "B_low" / f"{name}.txt")
            dups.append(f"./dup/images/B_low/{name}{f.suffix}")
        variants["B_low"] = (low + dups, f"A_low + {len(dups)} extra copies of low_keep gliomas")
        variants["C_low"] = (low + syn_low, f"A_low + {len(syn_low)} labeled synthetic gliomas")
    syn_full = add_syn("train_full")
    if syn_full:
        variants["C_full"] = (real + syn_full, f"A_full + {len(syn_full)} labeled synthetic gliomas")
        variants["D"] = (non_glioma + syn_full, f"real train without gliomas + {len(syn_full)} synthetic gliomas")

    meta = {}
    for name, (files, note) in variants.items():
        (stage / f"{name}.txt").write_text("\n".join(files) + "\n")
        n_gl = sum(1 for f in files if "/syn/" in f or "/dup/" in f or Path(f).name.startswith("gl_"))
        meta[name] = {"train_images": len(files), "glioma_images": n_gl, "note": note,
                      "list_sha1": hashlib.sha1("\n".join(files).encode()).hexdigest()[:12]}
        print(f"{name:7s} {len(files):4d} train images, {n_gl:3d} with glioma - {note}")
    (stage / "variants.json").write_text(json.dumps(meta, indent=1))

    zpath = OUT / "rq2_data.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT))
    shutil.rmtree(stage)
    print(f"{zpath}  ({zpath.stat().st_size / 1e6:.1f} MB)")
    # pieces under 10 MB for browser uploads; the notebook joins rq2_data.zip.partNN in name order
    for old in OUT.glob("rq2_data.zip.part*"):
        old.unlink()
    data, size = zpath.read_bytes(), 9_000_000
    for i in range(0, len(data), size):
        part = OUT / f"rq2_data.zip.part{i // size + 1:02d}"
        part.write_bytes(data[i:i + size])
        print(f"  {part.name}  {len(data[i:i + size]) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
