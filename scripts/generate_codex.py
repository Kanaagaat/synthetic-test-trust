"""Generate synthetic glioma MRI slices through the Codex CLI image tool.

Each image is one `codex exec` call with reference image(s) attached. Codex saves
the image under ~/.codex/generated_images/<thread_id>/ and logs the generation
(revised prompt, input image hashes) in its session rollout; this script collects
both, keeps the raw PNG, writes a grayscale 512x512 JPG and appends a row to
generated/manifest.csv.

Pilot v2 (one reference per image, 2 variants per reference):
    python scripts/generate_codex.py --prompt prompts/glioma_t1c_v2.md \
        --refs-json refs/pilot_v2/refs.json --per-ref 2 --out generated/pilot_v2 \
        --purpose pilot_v2 --prefix syn_gl_v2

Re-collect metadata for existing manifest rows (after changing the columns):
    python scripts/generate_codex.py --migrate-manifest
"""
import argparse
import csv
import fcntl
import os
import hashlib
import json
import random
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "generated" / "manifest.csv"
CODEX_HOME = Path.home() / ".codex"
SIZE = 512
STOP = threading.Event()  # set when OpenAI reports a usage limit

PLANES = ["axial", "sagittal", "coronal"]
LOCATIONS = [
    "in the left frontal lobe white matter",
    "in the right frontal lobe",
    "in the left temporal lobe",
    "in the right parietal lobe",
    "in the left parietal-occipital region",
    "deep near the right lateral ventricle",
]
SIZES = {
    "small": "small, about 8% of the image width",
    "medium": "medium, about 13% of the image width",
    "large": "large, about 20% of the image width",
}
PATTERNS_V1 = {
    "ring": "irregular thick ring of enhancement with a dark necrotic center and darker surrounding edema",
    "heterogeneous": "heterogeneous patchy enhancement with blurred, infiltrative margins",
    "cystic": "cystic mass with a thin enhancing rim and a small enhancing nodule",
    "low_grade": "faint minimal enhancement, mostly hypointense, poorly defined borders (low-grade look)",
}
PATTERNS_V2 = {
    **PATTERNS_V1,
    "low_grade": ("non-enhancing or only faintly enhancing ill-defined mass that is clearly darker than "
                  "the surrounding white matter, with visible mild mass effect such as effaced sulci or "
                  "a slightly compressed ventricle"),
}
ACQUISITIONS = {
    "glioma_t1c_v2": [
        "older 1.5T scanner: lower resolution, visible noise, softer contrast",
        "modern 3T scanner: sharp, low noise",
        "darker window, overall dim image",
        "brighter window with higher contrast",
        "head small in the frame with wide black margins",
        "head slightly off-center and filling most of the frame",
    ],
    # v3: pilot v2 showed "3T sharp" and "brighter window" give 5-10x the edge variance
    # of real images, and "head small" gives a head fraction of 0.13-0.17 (real 0.47 +- 0.10)
    "glioma_t1c_v3": [
        "older 1.5T scanner: lower resolution, visible noise, softer contrast",
        "standard clinical scanner, slightly soft image, moderate noise",
        "darker window, overall dim image",
        "slightly brighter window, soft image",
        "head slightly off-center and filling most of the frame",
        "head centered with moderate black margins, slightly soft image",
    ],
}
# v4: same variety as v3; softness is now requested for every image in the prompt itself
ACQUISITIONS["glioma_t1c_v4"] = ACQUISITIONS["glioma_t1c_v3"]
WRAPPER = (
    "Use your image generation tool to create exactly one image from the description below. "
    "Do not run shell commands, write files or ask questions. "
    "After the image is generated, reply only with the word done.\n\n"
)
FIELDS = [
    "file", "raw_file", "purpose", "generation_mode", "plane", "location", "size", "pattern",
    "acquisition", "refs", "reference_sha256", "prompt_version", "revised_prompt",
    "image_model", "orchestrator", "created_at", "codex_thread", "generation_id", "seconds",
    "status", "quality_score", "rejection_reason", "box_checked",
]


def load_template(path: Path) -> str:
    block = path.read_text().split("## Prompt", 1)[1]
    return re.search(r"```\n(.*?)```", block, re.S).group(1).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def codex_version() -> str:
    out = subprocess.run(["codex", "--version"], capture_output=True, text=True).stdout
    return out.strip().split()[-1] if out else "unknown"


def c2pa_agent(png: Path) -> str:
    """Generator name/version from the C2PA manifest OpenAI embeds in the PNG."""
    data = png.read_bytes()

    def cbor_text_after(key: bytes, start: int):
        # CBOR short text string: header byte 0x60 + length, then the bytes
        i = data.find(key, start)
        if i < 0:
            return None, -1
        j = i + len(key)
        n = data[j] - 0x60
        return (data[j + 1:j + 1 + n].decode(errors="ignore"), j + 1 + n) if 0 < n < 24 else (None, -1)

    name, pos = cbor_text_after(b"dname", max(data.find(b"softwareAgent"), 0))
    version, _ = cbor_text_after(b"gversion", pos) if pos > 0 else (None, -1)
    return f"{name}/{version}" if name else "unknown"


def codex_metadata(thread: str) -> dict:
    """Pull the image generation record and the orchestrating model from the rollout log."""
    meta = {"revised_prompt": "", "generation_id": "", "reference_sha256": "", "orchestrator_model": "",
            "saved_path": ""}
    logs = list((CODEX_HOME / "sessions").rglob(f"rollout-*{thread}.jsonl"))
    if not logs:
        return meta
    for line in logs[0].read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = ev.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        if not meta["orchestrator_model"] and isinstance(payload.get("model"), str):
            meta["orchestrator_model"] = payload["model"]
        item = payload.get("item") or {}
        if isinstance(item, dict) and item.get("kind") == "image_gen.generation":
            meta["revised_prompt"] = item.get("revisedPrompt", "")
            meta["generation_id"] = item.get("generationId") or item.get("id", "")
            meta["reference_sha256"] = ";".join(i.get("sha256", "") for i in item.get("inputImages", []))
            meta["saved_path"] = item.get("savedPath", "")
    return meta


def to_dataset_image(raw: Path, dst: Path):
    im = Image.open(raw).convert("L")
    side = max(im.size)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    canvas.resize((SIZE, SIZE), Image.LANCZOS).save(dst, quality=95)


def with_fill(spec, patterns):
    spec["fill"] = {"plane": spec["plane"], "location": spec["location"], "size": SIZES[spec["size"]],
                    "pattern": patterns[spec["pattern"]], "acquisition": spec["acquisition"]}
    return spec


def make_specs_v1(n, seed, prefix, start, refs):
    rng = random.Random(seed)
    specs = []
    for i in range(n):
        specs.append(with_fill({
            "name": f"{prefix}_{start + i:04d}",
            "plane": PLANES[i % len(PLANES)],
            "location": rng.choice(LOCATIONS),
            "size": rng.choices(list(SIZES), weights=[0.25, 0.5, 0.25])[0],
            "pattern": rng.choice(list(PATTERNS_V1)),
            "acquisition": "",
            "refs": refs,
        }, PATTERNS_V1))
    return specs


def make_specs_per_ref(refs_json: Path, per_ref, seed, prefix, start, acquisitions):
    rng = random.Random(seed)
    ref_dir = refs_json.parent
    specs, i = [], start
    for r in json.loads(refs_json.read_text()):
        for _ in range(per_ref):
            specs.append(with_fill({
                "name": f"{prefix}_{i:04d}",
                "plane": r["plane"],
                "location": rng.choice(LOCATIONS),
                "size": r["size"],
                "pattern": r["ref_pattern"],
                "acquisition": rng.choice(acquisitions),
                "refs": [ref_dir / r["ref"]],
            }, PATTERNS_V2))
            i += 1
    return specs


SUBSET = (ROOT / "data" / "subset_v1" if (ROOT / "data" / "subset_v1").exists() else ROOT.parent / "bonus_task" / "data" / "subset_v1")


def make_specs_pool(pool: str, n, per_ref, seed, prefix, start, acquisitions, only=None):
    """v3: references sampled from refs/pools.json; tumor type and plane come from the reference,
    size from the reference's largest real box."""
    files = json.loads((ROOT / "refs" / "pools.json").read_text())[pool]
    split = "test" if pool == "S_test_refs" else "train"
    rng = random.Random(seed)
    chosen = rng.sample(files, n) if n and n < len(files) else list(files)
    if only:  # regenerate specific references (e.g. twins rejected in review)
        chosen = [f for f in files if f in only]
    specs, i = [], start
    for f in chosen:
        boxes = [[float(v) for v in l.split()[1:5]]
                 for l in (SUBSET / "labels" / split / (Path(f).stem + ".txt")).read_text().splitlines() if l.strip()]
        w = max(boxes, key=lambda b: b[2] * b[3])[2]
        for _ in range(per_ref):
            acq = rng.choice(acquisitions)
            specs.append({
                "name": f"{prefix}_{i:04d}",
                "plane": "from_reference", "location": "different_from_reference",
                "size": f"{round(w * 100)}%", "pattern": f"from_reference;n={len(boxes)}", "acquisition": acq,
                "refs": [SUBSET / "images" / split / f],
                "fill": {"size_pct": max(3, round(w * 100)), "acquisition": acq,
                         "count_line": "Exactly one glioma" if len(boxes) == 1
                         else f"Exactly {len(boxes)} separate gliomas"},
                "n_tumors": len(boxes),
            })
            i += 1
    return specs


def run_one(spec, template, out: Path, prompt_version, orchestrator_fallback):
    prompt = template.format(**spec["fill"])
    cmd = ["codex", "exec", "--json", "--skip-git-repo-check", "-s", "read-only",
           "-C", str(ROOT), "-c", 'model_reasoning_effort="low"']
    for r in spec["refs"]:
        cmd += ["-i", str(r)]
    # `-i` takes several values, so a positional prompt would be read as an image path;
    # the prompt goes through stdin instead.
    t0 = time.time()
    proc = subprocess.run(cmd, input=WRAPPER + prompt, capture_output=True, text=True, timeout=900)
    secs = round(time.time() - t0)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "logs" / f"{spec['name']}.jsonl").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)

    thread = None
    for line in proc.stdout.splitlines():
        try:
            thread = thread or json.loads(line).get("thread_id")
        except json.JSONDecodeError:
            pass
    meta = codex_metadata(thread) if thread else {}
    pngs = sorted((CODEX_HOME / "generated_images" / thread).glob("*.png"),
                  key=lambda p: p.stat().st_mtime) if thread else []
    if not pngs:
        limit = re.search(r"usage limit[^\"]*", proc.stdout)
        if limit:
            STOP.set()  # no point in sending the rest of the batch
            print(f"[fail] {spec['name']}: {limit.group(0)[:160]} -> stopping the batch", flush=True)
        else:
            print(f"[fail] {spec['name']} ({secs}s) thread={thread} rc={proc.returncode}", flush=True)
        return None

    raw = out / "raw" / f"{spec['name']}.png"
    raw.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pngs[-1], raw)
    dst = out / f"{spec['name']}.jpg"
    to_dataset_image(raw, dst)
    print(f"[ok] {spec['name']} ({secs}s) {spec['plane']}, {spec['size']}, {spec['pattern']}", flush=True)
    return {
        "file": str(dst.relative_to(ROOT)), "raw_file": str(raw.relative_to(ROOT)),
        "generation_mode": "reference_guided",
        "plane": spec["plane"], "location": spec["location"], "size": spec["size"],
        "pattern": spec["pattern"], "acquisition": spec["acquisition"],
        "refs": ";".join(Path(r).name for r in spec["refs"]),
        "reference_sha256": meta.get("reference_sha256") or ";".join(sha256(Path(r)) for r in spec["refs"]),
        "prompt_version": prompt_version, "revised_prompt": meta.get("revised_prompt", ""),
        "image_model": c2pa_agent(raw),
        "orchestrator": f"codex-cli {codex_version()} / {meta.get('orchestrator_model') or orchestrator_fallback}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "codex_thread": thread, "generation_id": meta.get("generation_id", ""), "seconds": secs,
        "status": "unreviewed", "quality_score": "", "rejection_reason": "", "box_checked": "",
    }


def read_manifest():
    return list(csv.DictReader(MANIFEST.open())) if MANIFEST.exists() else []


@contextmanager
def manifest_lock():
    """Cross-process lock for read-modify-write of the manifest: the generator and
    review_server.py may run at the same time and both rewrite the whole file."""
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST.with_suffix(".lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def write_manifest(rows):
    """Atomic: readers never see a half-written file."""
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    os.replace(tmp, MANIFEST)


def migrate_manifest(orchestrator_fallback):
    rows = read_manifest()
    for r in rows:
        meta = codex_metadata(r["codex_thread"]) if r.get("codex_thread") else {}
        r.setdefault("generation_mode", "reference_guided")
        r["generation_mode"] = r.get("generation_mode") or "reference_guided"
        r["revised_prompt"] = r.get("revised_prompt") or meta.get("revised_prompt", "")
        r["generation_id"] = r.get("generation_id") or meta.get("generation_id", "")
        r["reference_sha256"] = r.get("reference_sha256") or meta.get("reference_sha256", "")
        r["image_model"] = r.get("image_model") or c2pa_agent(ROOT / r["raw_file"])
        r["orchestrator"] = r.get("orchestrator") or f"codex-cli {codex_version()} / {meta.get('orchestrator_model') or orchestrator_fallback}"
        if r.get("status", "").startswith("rejected") and not r.get("rejection_reason"):
            r["rejection_reason"] = r["status"].replace("rejected_", "")
            r["status"] = "rejected"
    write_manifest(rows)
    print(f"migrated {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="prompts/glioma_t1c.md")
    ap.add_argument("--n", type=int, default=1, help="v1 mode: number of images")
    ap.add_argument("--refs", default="refs", help="v1 mode: folder of shared references")
    ap.add_argument("--refs-json", help="per-reference mode: list of {ref, plane, size, ref_pattern}")
    ap.add_argument("--per-ref", type=int, default=2)
    ap.add_argument("--only", help="pool mode: comma-separated reference file names to (re)generate")
    ap.add_argument("--pool", choices=["R_train", "low_keep", "S_test_refs"],
                    help="v3 mode: sample references from refs/pools.json (--n refs, 0 = all)")
    ap.add_argument("--out", default="generated/pilot")
    ap.add_argument("--purpose", default="pilot")
    ap.add_argument("--prefix", default="syn_gl")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="generate at most this many missing images")
    ap.add_argument("--orchestrator-model", default="gpt-6.1-sol",
                    help="fallback if the rollout log does not name the Codex model")
    ap.add_argument("--migrate-manifest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.migrate_manifest:
        migrate_manifest(args.orchestrator_model)
        return

    prompt_path = ROOT / args.prompt
    prompt_version = prompt_path.stem
    template = load_template(prompt_path)
    out = ROOT / args.out
    if args.pool:
        specs = make_specs_pool(args.pool, args.n, args.per_ref, args.seed, args.prefix, args.start,
                                ACQUISITIONS[prompt_version], set(args.only.split(",")) if args.only else None)
    elif args.refs_json:
        specs = make_specs_per_ref(ROOT / args.refs_json, args.per_ref, args.seed, args.prefix, args.start,
                                   ACQUISITIONS[prompt_version])
    else:
        specs = make_specs_v1(args.n, args.seed, args.prefix, args.start, sorted((ROOT / args.refs).glob("*.jpg")))
    specs = [s for s in specs if not (out / f"{s['name']}.jpg").exists()]  # resume safely
    if args.limit:
        specs = specs[: args.limit]

    if args.dry_run:
        for s in specs:
            print({**s, "refs": [Path(r).name for r in s["refs"]]})
        if specs:
            print(template.format(**specs[0]["fill"]))
        return

    lock = threading.Lock()

    def job(s):
        if STOP.is_set():
            return None
        r = run_one(s, template, out, prompt_version, args.orchestrator_model)
        if r:  # append right away, so a crash mid-batch never leaves images without manifest rows
            with lock, manifest_lock():
                write_manifest(read_manifest() + [{**r, "purpose": args.purpose}])
        return r

    with ThreadPoolExecutor(args.workers) as pool:
        new = list(pool.map(job, specs))
    print(f"done: {sum(r is not None for r in new)}/{len(specs)} generated -> {out}"
          + ("  (stopped at the OpenAI usage limit; re-run the same command later to continue)" if STOP.is_set() else ""))


if __name__ == "__main__":
    main()
