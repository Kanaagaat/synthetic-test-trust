#!/usr/bin/env python3
"""Second S-test from a second generator: one Gemini twin per real test glioma.

Each Gemini twin gets the same prompt (v4) and the same substitutions (acquisition, size,
number of tumors, reference) as the OpenAI twin of the same reference that is used in the
S-test, so the two S-tests differ only in the generator. Gemini images are made in the Gemini
web app (driven through Claude in Chrome, a new chat per image); this script prepares the
prompts, ingests the downloaded images and draws a comparison sheet.

    python3 scripts/gemini_s_test.py spec                 # spec.json + prompts/<name>.txt
    python3 scripts/gemini_s_test.py ingest NAME FILE [--chat ID] [--seconds N]
    python3 scripts/gemini_s_test.py sheet                # compare_sheet.jpg

Naming: syn_gl_sg_XXXX has the number of the OpenAI twin syn_gl_st_XXXX of the same reference
(for references whose first OpenAI twin was rejected and regenerated, the regenerated one:
0029 -> 0052, 0031 -> 0053, 0037 -> 0054). 0040 (OpenAI twin rejected in blind review, never
regenerated) gets a Gemini twin too.
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_codex import (ROOT, SUBSET, load_template, manifest_lock, read_manifest,  # noqa: E402
                            sha256, to_dataset_image, write_manifest)
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

OUT = ROOT / "generated" / "s_test_gemini"
TEMPLATE = ROOT / "prompts" / "glioma_t1c_v4.md"
WRAP = "Create exactly one image from the description below.\n\n"
PURPOSE = "s_test_gemini"
IMAGE_MODEL = ("Gemini app (Plus, chat model 4 Hydrogen); image model not exposed; "
               "C2PA: Google Generative AI + SynthID")
ORCHESTRATOR = "Claude in Chrome"


def twin_rows():
    """reference -> the OpenAI twin row used in the S-test (latest non-rejected; else latest)."""
    by_ref = {}
    for r in read_manifest():
        if r["purpose"] != "s_test":
            continue
        cur = by_ref.get(r["refs"])
        if cur is None or r["status"] != "rejected" or cur["status"] == "rejected":
            by_ref[r["refs"]] = r
    return by_ref


def cmd_spec(_):
    refs = json.loads((ROOT / "refs" / "pools.json").read_text())["S_test_refs"]
    twins = twin_rows()
    template = load_template(TEMPLATE)
    (OUT / "prompts").mkdir(parents=True, exist_ok=True)
    spec = []
    for ref in refs:
        t = twins[ref]
        boxes = [[float(v) for v in l.split()[1:5]]
                 for l in (SUBSET / "labels" / "test" / (Path(ref).stem + ".txt")).read_text().splitlines()
                 if l.strip()]
        w = max(boxes, key=lambda b: b[2] * b[3])[2]
        n = len(boxes)
        # the same rule as make_specs_pool; must reproduce what the OpenAI twin got
        assert t["size"] == f"{round(w * 100)}%" and t["pattern"] == f"from_reference;n={n}", (ref, t["file"])
        fill = {"size_pct": max(3, round(w * 100)), "acquisition": t["acquisition"],
                "count_line": "Exactly one glioma" if n == 1 else f"Exactly {n} separate gliomas"}
        twin = Path(t["file"]).stem
        name = twin.replace("syn_gl_st_", "syn_gl_sg_")
        prompt = WRAP + template.format(**fill)
        (OUT / "prompts" / f"{name}.txt").write_text(prompt)
        spec.append({"name": name, "ref": ref, "openai_twin": twin, "openai_status": t["status"],
                     "openai_rejection": t["rejection_reason"], "n_tumors": n, **fill,
                     "plane": t["plane"], "location": t["location"], "size": t["size"],
                     "pattern": t["pattern"], "prompt": prompt})
    (OUT / "spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False))
    print(f"{len(spec)} specs -> {OUT / 'spec.json'}")
    probe = ROOT / "generated" / "gemini_probe" / "prompt_syn_gl_st_0000.txt"
    if probe.exists():
        same = probe.read_text().strip() == spec[0]["prompt"][len(WRAP):].strip()
        print(f"prompt of {spec[0]['name']} == probe prompt: {same}")


def load_spec():
    return {s["name"]: s for s in json.loads((OUT / "spec.json").read_text())}


def cmd_ingest(a):
    spec = load_spec()[a.name]
    if a.prompt_sha:  # sha256 prefix of the text actually typed into the chat
        sent = hashlib.sha256(spec["prompt"].encode()).hexdigest()
        if not sent.startswith(a.prompt_sha):
            sys.exit(f"prompt sha mismatch for {a.name}: chat {a.prompt_sha}, spec {sent[:12]}")
    src = Path(a.file).expanduser()
    with Image.open(src) as im:  # must be a real image before it goes into the project
        im.verify()
    with Image.open(src) as im:
        fmt, size = im.format, im.size
    data = src.read_bytes()
    google_c2pa = b"c2pa" in data and b"Google" in data
    raw = OUT / "raw" / f"{a.name}.jpeg"
    if raw.exists() and not a.force:
        sys.exit(f"{raw} exists (use --force to replace)")
    raw.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), raw)
    dst = OUT / f"{a.name}.jpg"
    to_dataset_image(raw, dst)
    ref_path = SUBSET / "images" / "test" / spec["ref"]
    row = {
        "file": str(dst.relative_to(ROOT)), "raw_file": str(raw.relative_to(ROOT)),
        "purpose": PURPOSE, "generation_mode": "reference_guided",
        "plane": spec["plane"], "location": spec["location"], "size": spec["size"],
        "pattern": spec["pattern"], "acquisition": spec["acquisition"],
        "refs": spec["ref"], "reference_sha256": sha256(ref_path),
        "prompt_version": "glioma_t1c_v4", "revised_prompt": "",
        "image_model": IMAGE_MODEL, "orchestrator": a.orchestrator,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "codex_thread": "", "generation_id": a.chat or "", "seconds": a.seconds or "",
        "status": "unreviewed", "quality_score": "", "rejection_reason": "", "box_checked": "",
    }
    with manifest_lock():
        rows = [r for r in read_manifest() if not (r["purpose"] == PURPOSE and r["file"] == row["file"])]
        rows.append(row)
        write_manifest(rows)
    print(f"[ok] {a.name}: {fmt} {size[0]}x{size[1]}, {len(data) / 1e6:.2f} MB, "
          f"Google C2PA: {google_c2pa} -> {dst.relative_to(ROOT)}")


def cmd_sheet(_):
    spec = list(load_spec().values())
    tile, head, per_row = 240, 34, 2
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    except OSError:
        font = ImageFont.load_default()
    rows = (len(spec) + per_row - 1) // per_row
    W, H = per_row * 3 * tile + (per_row - 1) * 16, rows * (tile + head) + 40
    sheet = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(sheet)
    d.text((8, 10), "reference (real test) | OpenAI twin (S-test) | Gemini twin", fill=0, font=font)

    def put(path, x, y, label):
        if path.exists():
            im = Image.open(path).convert("L")
            im.thumbnail((tile, tile))
            sheet.paste(im, (x + (tile - im.width) // 2, y + head + (tile - im.height) // 2))
        else:
            d.rectangle([x + 4, y + head + 4, x + tile - 4, y + head + tile - 4], outline=128)
            d.text((x + 70, y + head + tile // 2), "missing", fill=0, font=font)
        d.text((x + 4, y + 4), label, fill=0, font=font)

    for i, s in enumerate(spec):
        x0 = (i % per_row) * (3 * tile + 16)
        y = 40 + (i // per_row) * (tile + head)
        num = s["name"][-4:]
        rej = " REJECTED" if s["openai_status"] == "rejected" else ""
        put(SUBSET / "images" / "test" / s["ref"], x0, y, f"{num} ref, n={s['n_tumors']}, {s['size_pct']}%")
        put(ROOT / "generated" / "s_test" / f"{s['openai_twin']}.jpg", x0 + tile, y, f"OpenAI st_{num}{rej}")
        put(OUT / f"{s['name']}.jpg", x0 + 2 * tile, y, f"Gemini sg_{num}")
    dst = OUT / "compare_sheet.jpg"
    sheet.save(dst, quality=88)
    print(f"-> {dst.relative_to(ROOT)} ({W}x{H})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("spec")
    p = sub.add_parser("ingest")
    p.add_argument("name")
    p.add_argument("file")
    p.add_argument("--chat", help="Gemini chat id (stored in generation_id; not exported)")
    p.add_argument("--seconds", type=int)
    p.add_argument("--prompt-sha", help="sha256 prefix of the prompt typed into the chat (checked against spec)")
    p.add_argument("--orchestrator", default=ORCHESTRATOR, help="who drove the Gemini app (and which account)")
    p.add_argument("--force", action="store_true")
    sub.add_parser("sheet")
    a = ap.parse_args()
    {"spec": cmd_spec, "ingest": cmd_ingest, "sheet": cmd_sheet}[a.cmd](a)


if __name__ == "__main__":
    main()
