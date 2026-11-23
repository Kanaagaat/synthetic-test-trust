"""Local box-review tool for synthetic images (standard library only + the manifest helpers).

    python3 scripts/review_server.py --purpose pilot_v2        # then open http://127.0.0.1:8765

Shows each image of <purpose> with the model's draft box (labels_draft/). You accept it, redraw it,
or reject the image. Results:
    generated/<purpose>/labels/<name>.txt   final YOLO label (class 0 = glioma)
    generated/<purpose>/review_log.csv      decision, draft vs final IoU, timestamp (one row per save)
    generated/manifest.csv                  status labeled/rejected, box_checked unchanged/corrected/drawn
Keys: drag = new box, click a box + Backspace = delete it, Enter = save, R = reject, <- / -> = navigate.
Serves on 127.0.0.1 only.

--blind: no model drafts are shown and boxes go to labels_blind/ (log review_log_blind.csv). Used for
the S-test, because boxes accepted from the bonus model's drafts would be anchored to the very model
being evaluated (an accepted draft has IoU 1.0 with that model's prediction).

--real-test: blind relabeling of the REAL test glioma images (refs/pools.json S_test_refs) by the same
person who labeled the twins, so real test and S-test can be compared with boxes from one annotator.
The dataset's own boxes are never shown. Output: generated/real_test_blind/labels_blind/ and
review_log_blind.csv; the manifest is not touched. "Reject" here means "I cannot see a tumor".
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_codex as g  # noqa: E402  (manifest read/write)

ROOT = g.ROOT


def read_boxes(path: Path):
    if not path.exists():
        return None
    return [[float(v) for v in l.split()[1:5]] for l in path.read_text().splitlines() if l.strip()]


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def reviewable(r, purpose):
    if r["purpose"] != purpose:
        return False
    if r["status"] in ("accepted_needs_box", "labeled"):
        return True
    return r["status"] == "rejected" and r["rejection_reason"].startswith("review:")


SUBSET = (ROOT / "data" / "subset_v1" if (ROOT / "data" / "subset_v1").exists() else ROOT.parent / "bonus_task" / "data" / "subset_v1")


class Handler(BaseHTTPRequestHandler):
    purpose = "pilot_v2"
    blind = False
    real_test = False

    def label_dir(self):
        return self.gen_dir() / ("labels_blind" if self.blind else "labels")

    def gen_dir(self):
        return ROOT / "generated" / ("real_test_blind" if self.real_test else self.purpose)

    def real_items(self):
        done = {}
        log = self.gen_dir() / "review_log_blind.csv"
        if log.exists():
            for r in csv.DictReader(log.open()):
                done[r["name"]] = r  # last decision wins
        items = []
        for f in json.loads((ROOT / "refs" / "pools.json").read_text())["S_test_refs"]:
            name = Path(f).stem
            n = sum(1 for l in (SUBSET / "labels" / "test" / f"{name}.txt").read_text().splitlines() if l.strip())
            last = done.get(name)
            status = "accepted_needs_box" if last is None else ("rejected" if last["decision"] == "reject" else "labeled")
            items.append({"name": name, "plane": "real test", "size": "", "pattern": f"n={n}", "status": status,
                          "box_checked": "blind" if status == "labeled" else "",
                          "reason": last["reason"] if status == "rejected" else "", "draft": [],
                          "final": read_boxes(self.label_dir() / f"{name}.txt")})
        return items

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.path = urlsplit(self.path).path  # drop ?t= cache busters
        if self.path == "/":
            banner = ('<p style="color:var(--warn);font-weight:600">Слепой режим: черновиков модели нет, '
                      'рисуй рамку сам с нуля.</p>') if self.blind else ""
            if self.real_test:
                banner += ('<p class="muted">Это <b>реальные</b> глиомы теста. Рамки авторов датасета не показываются. '
                           'Рисуй так же, как рисовал рамки двойникам. «Отбраковать» здесь значит «не вижу опухоль».</p>')
            page = PAGE.replace("__PURPOSE__", self.purpose + (" · blind" if self.blind else "")).replace("__BLIND__", banner)
            return self.send(200, page, "text/html; charset=utf-8")
        if self.path == "/api/items" and self.real_test:
            return self.send(200, json.dumps(self.real_items()))
        if self.path == "/api/items":
            items = []
            for r in g.read_manifest():
                if not reviewable(r, self.purpose):
                    continue
                if self.blind and r["status"] == "rejected" and not r["rejection_reason"].startswith("review(blind)"):
                    continue  # rejected in the assisted pass and replaced by a regenerated twin
                name = Path(r["file"]).stem
                final = read_boxes(self.label_dir() / f"{name}.txt")
                status = r["status"]
                if self.blind and status != "rejected":
                    status = "labeled" if final is not None else "accepted_needs_box"
                items.append({
                    "name": name, "plane": r["plane"], "size": r["size"], "pattern": r["pattern"],
                    "status": status,
                    "box_checked": ("blind" if final is not None else "") if self.blind else r["box_checked"],
                    "reason": r["rejection_reason"],
                    "draft": [] if self.blind else (read_boxes(self.gen_dir() / "labels_draft" / f"{name}.txt") or []),
                    "final": final,
                })
            return self.send(200, json.dumps(items))
        if self.path.startswith("/img/"):
            name = Path(self.path[5:]).name  # no path traversal
            f = (SUBSET / "images" / "test" if self.real_test else self.gen_dir()) / f"{name}.jpg"
            if f.exists():
                return self.send(200, f.read_bytes(), "image/jpeg")
        self.send(404, "not found", "text/plain")

    def do_POST(self):
        try:
            return self._post()
        except Exception as exc:  # report the error to the page instead of dropping the connection
            import traceback
            traceback.print_exc()
            return self.send(500, json.dumps({"error": f"ошибка сервера: {type(exc).__name__}: {exc}"}))

    def _post(self):
        if self.path != "/api/save":
            return self.send(404, "not found", "text/plain")
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        name = Path(req["name"]).name
        boxes = [[min(max(float(v), 0.0), 1.0) for v in b] for b in req.get("boxes", [])]
        decision = req["decision"]  # "save" | "reject"
        draft = [] if self.blind else (read_boxes(self.gen_dir() / "labels_draft" / f"{name}.txt") or [])

        if decision == "reject":
            checked, status = "", "rejected"
            reason = ("review(blind): " if self.blind else "review: ") + (req.get("reason") or "unspecified")
            (self.label_dir() / f"{name}.txt").unlink(missing_ok=True)
        else:
            if not boxes:
                return self.send(400, json.dumps({"error": "draw at least one box or reject"}))
            same = len(boxes) == len(draft) and all(max(iou(b, d) for d in draft) > 0.98 for b in boxes)
            checked = "blind" if self.blind else ("unchanged" if same else ("drawn" if not draft else "corrected"))
            status, reason = "labeled", ""
            self.label_dir().mkdir(parents=True, exist_ok=True)
            (self.label_dir() / f"{name}.txt").write_text(
                "".join("0 " + " ".join(f"{v:.6f}" for v in b) + "\n" for b in boxes))

        best = [max((iou(b, d) for d in draft), default=0.0) for b in boxes]
        self.gen_dir().mkdir(parents=True, exist_ok=True)
        log = self.gen_dir() / ("review_log_blind.csv" if self.blind else "review_log.csv")
        new = not log.exists()
        with log.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["name", "decision", "box_checked", "n_draft", "n_final", "draft_iou_mean", "reason", "time"])
            w.writerow([name, decision, checked, len(draft), len(boxes),
                        round(sum(best) / len(best), 3) if best else "", reason, datetime.now().isoformat(timespec="seconds")])

        if self.real_test:  # real images are not in the manifest
            return self.send(200, json.dumps({"ok": True, "status": status, "box_checked": checked}))
        with g.manifest_lock():  # the generator may be appending rows right now
            rows = g.read_manifest()
            for r in rows:
                if r["purpose"] == self.purpose and Path(r["file"]).stem == name:
                    if self.blind and decision == "save":
                        # the assisted label (if any) stays as it was; a twin labeled only blind becomes "labeled"
                        if r["status"] != "labeled":
                            r["status"], r["box_checked"], r["rejection_reason"] = "labeled", "blind", ""
                    else:
                        r["status"], r["box_checked"], r["rejection_reason"] = status, checked, reason
            g.write_manifest(rows)
        self.send(200, json.dumps({"ok": True, "status": status, "box_checked": checked}))


PAGE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Box review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#deded8;--ok:#1f8a4c;--warn:#b25b00;--bad:#b3261e;--accent:#2f5bd3}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--muted:#a3a39d;--card:#20201e;--line:#33332f;--ok:#4fc27f;--warn:#f0a040;--bad:#f07067;--accent:#7f9cf5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px;display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px}
@media (max-width:860px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
canvas{width:100%;max-width:640px;aspect-ratio:1;display:block;margin:0 auto;cursor:crosshair;background:#000;border-radius:6px}
h1{font-size:16px;margin:0 0 4px}.muted{color:var(--muted)}.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
button{font:inherit;padding:7px 12px;border-radius:7px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button.danger{color:var(--bad)}
.tag{display:inline-block;padding:1px 7px;border-radius:99px;border:1px solid var(--line);font-size:12px;margin-right:4px}
.labeled{color:var(--ok)}.rejected{color:var(--bad)}.accepted_needs_box{color:var(--warn)}
ol{padding-left:30px;margin:6px 0;max-height:360px;overflow:auto}li{cursor:pointer;padding:1px 0}li.cur{font-weight:600}
kbd{border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:12px}
#msg{min-height:20px;margin-top:8px;font-weight:600}[hidden]{display:none!important}
</style></head><body><main>
<section class="card"><h1 id="title">…</h1><div id="meta" class="muted"></div>
<canvas id="cv" width="640" height="640"></canvas>
<div class="row"><button class="primary" id="save">Сохранить рамку <kbd>Enter</kbd></button>
<button id="reset">Вернуть черновик</button><button id="clear">Стереть всё</button>
<button class="danger" id="reject">Отбраковать <kbd>R</kbd></button></div>
<div id="rejbox" class="row" hidden><span class="muted">Причина:</span>
<button class="danger rr">опухоль не видна</button><button class="danger rr">анатомия неправдоподобна</button>
<button class="danger rr">артефакт / текст / цвет</button>
<input id="rejother" placeholder="другое — Enter" style="font:inherit;padding:6px 8px;border-radius:7px;border:1px solid var(--line);background:var(--card);color:var(--fg)">
<button id="rejcancel">Отмена</button></div>
<div id="msg" class="muted"></div></section>
<aside class="card"><h1>__PURPOSE__</h1>__BLIND__<div id="progress" class="muted"></div>
<p class="muted">Оранжевая пунктирная — черновик модели, зелёная — твоя рамка. Тяни мышью, чтобы нарисовать новую. Клик по рамке + <kbd>⌫</kbd> — удалить. <kbd>←</kbd> <kbd>→</kbd> — листать.</p>
<p class="muted">Если клавиши не срабатывают, сначала кликни по картинке. После каждого сохранения внизу появляется зелёная строка «✓ сохранено» — без неё сохранения не было.</p>
<p class="muted">Рамка должна охватывать всю опухоль (включая кольцо накопления), но не отёк вокруг — как в реальной разметке датасета.</p>
<ol id="list"></ol></aside></main>
<script>
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');let items=[],i=0,boxes=[],sel=-1,img=new Image(),drag=null;
const $=id=>document.getElementById(id);
async function load(){items=await (await fetch('/api/items')).json();const f=items.findIndex(x=>x.status==='accepted_needs_box');i=f<0?0:f;show()}
function show(){if(!items.length){$('title').textContent='Нечего проверять';return}
 const it=items[i];boxes=(it.final??it.draft).map(b=>b.slice());sel=-1;
 $('title').textContent=`${it.name}  (${i+1}/${items.length})`;
 $('meta').innerHTML=`<span class="tag">${it.plane}</span><span class="tag">${it.size}</span><span class="tag">${it.pattern}</span> <span class="${it.status}">${it.status}${it.box_checked?' · '+it.box_checked:''}${it.reason?' · '+it.reason:''}</span>`+(it.draft.length?'':' · <b>черновика нет, нарисуй рамку</b>');
 img=new Image();img.onload=draw;img.src='/img/'+it.name+'?t='+Date.now();renderList();$('msg').textContent=''}
function renderList(){const done=items.filter(x=>x.status!=='accepted_needs_box').length;$('progress').textContent=`Проверено ${done} из ${items.length}`;
 $('list').innerHTML=items.map((x,k)=>`<li class="${k===i?'cur':''} ${x.status}" data-k="${k}">${x.name.slice(-4)} — ${x.status==='labeled'?'✓ '+x.box_checked:x.status==='rejected'?'✗':'…'}</li>`).join('');
 [...$('list').children].forEach(li=>li.onclick=()=>{i=+li.dataset.k;show()})}
function rect(b){const S=cv.width;return[(b[0]-b[2]/2)*S,(b[1]-b[3]/2)*S,b[2]*S,b[3]*S]}
function draw(){const S=cv.width;ctx.clearRect(0,0,S,S);ctx.drawImage(img,0,0,S,S);const it=items[i];
 ctx.setLineDash([6,4]);ctx.lineWidth=2;ctx.strokeStyle='#ff9f1a';it.draft.forEach(b=>ctx.strokeRect(...rect(b)));ctx.setLineDash([]);
 boxes.forEach((b,k)=>{ctx.lineWidth=k===sel?4:2;ctx.strokeStyle='#38d26b';ctx.strokeRect(...rect(b))});
 if(drag&&drag.w){ctx.strokeStyle='#7fb2ff';ctx.lineWidth=2;ctx.strokeRect(drag.x,drag.y,drag.w,drag.h)}}
function pos(e){const r=cv.getBoundingClientRect();return[(e.clientX-r.left)*cv.width/r.width,(e.clientY-r.top)*cv.height/r.height]}
cv.onmousedown=e=>{const[x,y]=pos(e);drag={x,y,w:0,h:0,x0:x,y0:y}};
cv.onmousemove=e=>{if(!drag)return;const[x,y]=pos(e);drag.x=Math.min(x,drag.x0);drag.y=Math.min(y,drag.y0);drag.w=Math.abs(x-drag.x0);drag.h=Math.abs(y-drag.y0);draw()};
cv.onmouseup=e=>{const S=cv.width;if(drag&&drag.w>4&&drag.h>4){boxes.push([(drag.x+drag.w/2)/S,(drag.y+drag.h/2)/S,drag.w/S,drag.h/S]);sel=boxes.length-1}
 else if(drag){const[x,y]=[drag.x0/S,drag.y0/S];sel=boxes.findIndex(b=>Math.abs(x-b[0])<=b[2]/2&&Math.abs(y-b[1])<=b[3]/2)}drag=null;draw()};
let busy=false;
function say(text,color){const m=$('msg');m.textContent=text;m.style.color=color||''}
async function save(decision,reason){if(busy)return;busy=true;const it=items[i];
 try{const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:it.name,boxes,decision,reason})});
  const j=await r.json();if(!r.ok){say('✗ '+(j.error||'ошибка сохранения'),'var(--bad)');return}
  it.status=j.status;it.box_checked=j.box_checked;it.final=decision==='reject'?null:boxes.map(b=>b.slice());it.reason=decision==='reject'?'review: '+reason:'';
  $('rejbox').hidden=true;
  const nxt=items.findIndex((x,k)=>k>i&&x.status==='accepted_needs_box');if(nxt>=0){i=nxt}show();
  say(`✓ ${it.name.slice(-4)} сохранено (${decision==='reject'?'отбраковано: '+reason:j.box_checked})`+(nxt>=0?'':' · непроверенных дальше нет'),'var(--ok)')}
 catch(err){say('✗ сервер не отвечает — сохранение не прошло','var(--bad)')}
 finally{busy=false}}
$('save').onclick=()=>save('save');$('reset').onclick=()=>{boxes=items[i].draft.map(b=>b.slice());sel=-1;draw()};$('clear').onclick=()=>{boxes=[];sel=-1;draw()};
$('reject').onclick=()=>{$('rejbox').hidden=false};$('rejcancel').onclick=()=>{$('rejbox').hidden=true};
document.querySelectorAll('.rr').forEach(b=>b.onclick=()=>save('reject',b.textContent));
$('rejother').onkeydown=e=>{if(e.key==='Enter'&&e.target.value.trim()){save('reject',e.target.value.trim());e.target.value=''}};
document.onkeydown=e=>{if(e.target.tagName==='INPUT')return;if(e.key==='Enter'){e.preventDefault();save('save')}else if(e.key==='r'||e.key==='R'||e.key==='к'||e.key==='К'){$('reject').click()}
 else if(e.key==='ArrowRight'){i=Math.min(i+1,items.length-1);show()}else if(e.key==='ArrowLeft'){i=Math.max(i-1,0);show()}
 else if((e.key==='Backspace'||e.key==='Delete')&&sel>=0){boxes.splice(sel,1);sel=-1;draw()}};
load();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--purpose", default="pilot_v2")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--blind", action="store_true", help="no model drafts; save to labels_blind/")
    ap.add_argument("--real-test", action="store_true", help="blind relabel of the real test gliomas")
    args = ap.parse_args()
    Handler.purpose = "real_test_blind" if args.real_test else args.purpose
    Handler.blind = args.blind or args.real_test
    Handler.real_test = args.real_test
    print(f"review {args.purpose}: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
