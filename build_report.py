#!/usr/bin/env python3
"""
Build a self-contained HTML evidence report from benchmark output.

Reads outputs/v<model>/results.json and renders a tabbed report: a summary tab
plus one tab per category, each with per-case chips and an Object Classes /
Parsed Text panel.

Reference data — prior-version output, verdicts, assessments, disclaimers — is
optional and lives in a separate file that is NOT committed, so the harness can
be published without publishing an evaluation. See reference/SCHEMA.md.

Usage
-----
  python3.10 build_report.py --model 2.0
  python3.10 build_report.py --model 2.0 --reference reference/reference.json
  python3.10 build_report.py --model 2.0 --reference reference/reference.json \\
                             --screenshots reference/screenshots.json
  python3.10 build_report.py --model 2.0 --out report/evidence.html

Without --reference the report still renders: every case shows its parsed
output, latency and source link, with no comparison column and no verdicts.
"""

import argparse
import base64
import html
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------

VERDICTS = {
    "fixed": ("t-fix", "Fixed", "\u2713"),
    "part":  ("t-part", "Partial", "~"),
    "open":  ("t-open", "Not fixed", "\u00b7"),
    "reg":   ("t-reg", "Regressed", "!"),
}
DEFAULT_VERDICT = ("t-none", "Not assessed", "\u00b7")


def esc(s):
    return html.escape(str(s or ""))


def load_results(out_root, model):
    p = out_root / f"v{model}" / "results.json"
    if not p.exists():
        sys.exit(f"No results at {p}\nRun:  python3.10 nemotron_eval.py --model {model}")
    return json.loads(p.read_text())


def load_json(path, what):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        sys.exit(f"{what} not found: {p}")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------

def oc_panel(rows, highlight_cls=None):
    """Object Classes / Parsed Text table."""
    out = ['<table class="oc"><thead><tr><th>Object Classes</th>'
           '<th>Parsed Text</th></tr></thead><tbody>']
    for cls, txt in rows:
        mark = ' class="hl-row"' if highlight_cls and str(cls).lower() == highlight_cls.lower() else ""
        body = esc(txt)[:340] or "<i>&mdash;</i>"
        out.append(f'<tr{mark}><td class="oc-c">{esc(cls)}</td>'
                   f'<td class="oc-t">{body}</td></tr>')
    return "".join(out) + "</tbody></table>"


def build_case(rec, ref, shots):
    ex = rec["example"]
    eid = ex["eid"]
    r = ref.get(eid, {})

    vkey = r.get("verdict")
    vcls, vlabel, _ = VERDICTS.get(vkey, DEFAULT_VERDICT)

    ours = [(e["cls"], (e.get("text") or "").replace("\n", " "))
            for e in rec.get("elements", [])]
    prior = r.get("prior_output")
    disc = r.get("disclaimer")
    hl = disc.get("highlight_class") if disc else None

    lat = rec.get("latency", {})
    shot = shots.get(eid)

    # left column only when reference output is supplied
    if prior:
        left = (f'<div><div class="lab">{esc(r.get("prior_label", "Prior version"))}</div>'
                f'{oc_panel([(c, t) for c, t in prior])}</div>')
        grid = "two"
    else:
        left, grid = "", "one"

    assess = (f'<div class="why"><span class="wl">Assessment</span>'
              f'{esc(r["assessment"])}</div>') if r.get("assessment") else ""

    disc_html = ""
    if disc:
        disc_html = (f'<div class="disc"><div class="dh">{esc(disc.get("heading", "Disclaimer"))}'
                     f'<span class="dtag">{esc(disc.get("kind", ""))}</span></div>'
                     f'<p>{esc(disc.get("note", ""))}</p></div>')

    shot_html = ""
    if shot:
        shot_html = (f'<div class="sv" id="v-{eid}-shot">'
                     f'<div class="lab">Reference screenshot '
                     f'<span class="prov">{esc(shot.get("src", ""))}</span></div>'
                     f'<img src="data:image/jpeg;base64,{shot["b64"]}" '
                     f'alt="Reference screenshot for {esc(ex["label"])}"></div>')
        tabs = (f'<div class="subtabs">'
                f'<button class="st active" data-c="{eid}" data-v="cmp">Parsed text</button>'
                f'<button class="st" data-c="{eid}" data-v="shot">Reference screenshot</button>'
                f'</div>')
    else:
        tabs = ""

    issue = f'<div class="ci">{esc(r["issue"])}</div>' if r.get("issue") else ""
    tag = f'<span class="tag {vcls}">{vlabel}</span>' if vkey else ""

    return f'''<div class="case">
 <div class="case-h"><div><div class="ct">{esc(ex["label"])}</div>{issue}</div>{tag}</div>
 {assess}{disc_html}{tabs}
 <div class="sv active" id="v-{eid}-cmp"><div class="{grid}">{left}
   <div><div class="lab">Parse {rec.get("version", "")} output
     <span class="n">{rec.get("n_elements", 0)} elements &middot; {lat.get("total_s", "?")}s</span></div>
     {oc_panel(ours, hl)}</div></div></div>
 {shot_html}
 <div class="src"><a href="{esc(ex["url"])}#page={ex["page"]}" target="_blank" rel="noopener">
   &#8599; source PDF, page {ex["page"]}</a>
   <span class="url">{esc(ex["url"])}#page={ex["page"]}</span></div>
</div>'''


def stat_block(cells):
    return '<div class="stats">' + "".join(
        f'<div class="stat {k}"><div class="sn">{v}</div><div class="sl">{l}</div></div>'
        for l, v, k in cells) + "</div>"


def build(records, ref, shots, model):
    cats = {}
    for rec in records:
        cats.setdefault(rec["example"]["cat_name"], []).append(rec)
    cats = dict(sorted(cats.items()))

    def counts(recs):
        c = {k: 0 for k in VERDICTS}
        c["disc"] = 0
        for r in recs:
            e = ref.get(r["example"]["eid"], {})
            if e.get("verdict") in c:
                c[e["verdict"]] += 1
            if e.get("disclaimer"):
                c["disc"] += 1
        return c

    total = counts(records)
    has_ref = any(ref.get(r["example"]["eid"], {}).get("verdict") for r in records)

    def cells_for(c, n):
        if not has_ref:
            lat = [r["latency"]["total_s"] for r in records if r.get("latency", {}).get("total_s")]
            el = sum(r.get("n_elements", 0) for r in records)
            return [("Pages tested", n, ""), ("Elements found", el, ""),
                    ("Mean latency", f'{sum(lat)/len(lat):.2f}s' if lat else "—", ""),
                    ("Model", f"v{model}", "")]
        return [("Pages tested", n, ""),
                ("Fixed", c["fixed"], "s-fix"),
                ("Partial", c["part"], "s-part"),
                ("Not fixed", c["open"] + c["reg"], "s-open"),
                ("With disclaimer", c["disc"], "s-disc")]

    # nav
    nav = [f'<button class="nav-i active" data-t="summary"><span class="ni">Summary</span>'
           f'<span class="nc">{len(records)} pages</span></button>']
    for cat, recs in cats.items():
        c = counts(recs)
        flag = f'<span class="dotd">{c["disc"]}</span>' if c["disc"] else ""
        extra = f'{c["fixed"]} fixed ' if has_ref else ""
        nav.append(f'<button class="nav-i" data-t="{esc(cat)}"><span class="ni">'
                   f'{esc(cat.replace("_", " ").title())}</span>'
                   f'<span class="nc">{len(recs)} pages · {extra}{flag}</span></button>')

    # summary table
    rows = []
    for cat, recs in cats.items():
        c = counts(recs)
        if has_ref:
            rows.append(f'<tr><td>{esc(cat.replace("_", " ").title())}</td>'
                        f'<td class="r">{len(recs)}</td><td class="r ok">{c["fixed"] or "—"}</td>'
                        f'<td class="r">{c["part"] or "—"}</td>'
                        f'<td class="r">{(c["open"]+c["reg"]) or "—"}</td>'
                        f'<td class="r bad">{c["disc"] or "—"}</td></tr>')
        else:
            el = sum(r.get("n_elements", 0) for r in recs)
            lat = [r["latency"]["total_s"] for r in recs if r.get("latency", {}).get("total_s")]
            rows.append(f'<tr><td>{esc(cat.replace("_", " ").title())}</td>'
                        f'<td class="r">{len(recs)}</td><td class="r">{el}</td>'
                        f'<td class="r">{sum(lat)/len(lat):.2f}s</td></tr>')

    thead = ('<tr><th>Category</th><th class="r">Pages</th><th class="r">Fixed</th>'
             '<th class="r">Partial</th><th class="r">Not fixed</th>'
             '<th class="r">Disclaimer</th></tr>') if has_ref else \
            ('<tr><th>Category</th><th class="r">Pages</th><th class="r">Elements</th>'
             '<th class="r">Mean latency</th></tr>')

    lat_all = sorted(r["latency"]["total_s"] for r in records
                     if r.get("latency", {}).get("total_s"))
    tps = [r["latency"]["tokens_per_sec"] for r in records
           if r.get("latency", {}).get("tokens_per_sec")]
    meta = records[0]
    kv = [("Model", meta.get("model", "")),
          ("Pages", len(records)),
          ("Elements", sum(r.get("n_elements", 0) for r in records)),
          ("Latency", f"median {lat_all[len(lat_all)//2]:.2f}s · "
                      f"p90 {lat_all[int(len(lat_all)*0.9)]:.2f}s · "
                      f"max {lat_all[-1]:.2f}s" if lat_all else "—"),
          ("Throughput", f"{sum(tps)/len(tps):.0f} tok/s mean" if tps else "—"),
          ("Image source", meta.get("image", {}).get("source", ""))]

    summary = (f'<div class="hdr"><h1>Summary</h1>'
               f'<p class="sub">Nemotron Parse {model} — {len(records)} pages</p></div>'
               + stat_block(cells_for(total, len(records)))
               + f'<table class="sum"><thead>{thead}</thead><tbody>{"".join(rows)}</tbody></table>'
               + '<div class="kv">' + "".join(
                   f'<div><span>{esc(k)}</span>{esc(v)}</div>' for k, v in kv) + '</div>')

    panes = [f'<section class="pane active" id="p-summary">{summary}</section>']
    for cat, recs in cats.items():
        c = counts(recs)
        chips, cases = [], []
        for i, rec in enumerate(recs):
            eid = rec["example"]["eid"]
            e = ref.get(eid, {})
            _, _, glyph = VERDICTS.get(e.get("verdict"), DEFAULT_VERDICT)
            d = '<span class="cd">!</span>' if e.get("disclaimer") else ""
            chips.append(f'<button class="chip{" active" if i==0 else ""}" data-cat="{esc(cat)}" '
                         f'data-c="{eid}"><span class="cm {e.get("verdict","none")}">{glyph}</span>'
                         f'{esc(rec["example"]["label"])}{d}</button>')
            cases.append(f'<div class="cw{" active" if i==0 else ""}" id="w-{esc(cat)}-{eid}">'
                         f'{build_case(rec, ref, shots)}</div>')
        panes.append(f'<section class="pane" id="p-{esc(cat)}">'
                     f'<div class="hdr"><h1>{esc(cat.replace("_"," ").title())}</h1></div>'
                     f'{stat_block(cells_for(c, len(recs)))}'
                     f'<div class="chips">{"".join(chips)}</div>{"".join(cases)}</section>')

    return f'<nav id="nav">{"".join(nav)}</nav><main id="main">{"".join(panes)}</main>'


CSS = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nemotron Parse — evaluation report</title><style>
:root{--ink:#0b0b0b;--ink2:#52514e;--ink3:#898781;--line:#e1e0d9;--line2:#c3c2b7;
--surf:#fff;--surf1:#fcfcfb;--surf0:#f7f6f3;--green:#0F6E56;--greenbg:#E1F5EE;
--red:#B3261E;--redbg:#FAECE7;--amber:#854F0B;--amberbg:#FAEEDA;--blue:#2a78d6;--dark:#501313}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
background:var(--surf0);color:var(--ink);line-height:1.55;display:flex;min-height:100vh}
#nav{width:236px;flex:0 0 236px;background:var(--surf);border-right:1px solid var(--line2);
padding:18px 0;position:sticky;top:0;height:100vh;overflow-y:auto}
.nav-i{display:block;width:100%;text-align:left;background:none;border:none;cursor:pointer;
padding:10px 18px;border-left:3px solid transparent;font-family:inherit}
.nav-i:hover{background:var(--surf1)}
.nav-i.active{background:var(--surf0);border-left-color:var(--blue)}
.ni{display:block;font-size:13px;font-weight:600;color:var(--ink)}
.nc{display:block;font-size:11px;color:var(--ink3);margin-top:1px}
.dotd{display:inline-block;background:var(--dark);color:#fff;font-size:10px;font-weight:600;
padding:0 5px;border-radius:100px;margin-left:3px}
#main{flex:1;min-width:0;padding:28px 30px 60px;max-width:1080px}
.pane{display:none}.pane.active{display:block}
.hdr{margin-bottom:16px;padding-bottom:13px;border-bottom:1px solid var(--line2)}
h1{font-size:22px;font-weight:600;letter-spacing:-.01em}
.sub{font-size:13px;color:var(--ink2);margin-top:4px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:9px;margin-bottom:16px}
.stat{background:var(--surf);border:1px solid var(--line);border-radius:9px;padding:11px 13px}
.sn{font-size:22px;font-weight:600;letter-spacing:-.02em}
.sl{font-size:11px;color:var(--ink3);line-height:1.35}
.stat.s-fix .sn{color:var(--green)}.stat.s-part .sn{color:var(--amber)}
.stat.s-disc{background:var(--redbg);border-color:#F0997B}.stat.s-disc .sn{color:var(--red)}
table.sum{width:100%;border-collapse:collapse;background:var(--surf);border:1px solid var(--line);
border-radius:10px;overflow:hidden;margin-bottom:16px}
table.sum th{font-size:11px;font-weight:600;color:var(--ink2);text-align:left;padding:9px 12px;
background:var(--surf1);border-bottom:1px solid var(--line2)}
table.sum th.r,table.sum td.r{text-align:right}
table.sum td{font-size:13px;padding:9px 12px;border-bottom:1px solid var(--line)}
table.sum tr:last-child td{border-bottom:none}
td.ok{color:var(--green);font-weight:600}td.bad{color:var(--red);font-weight:600}
.kv{background:var(--surf);border:1px solid var(--line);border-radius:10px;padding:4px 15px}
.kv div{font-size:12px;color:var(--ink2);padding:6px 0;border-bottom:1px solid var(--line);display:flex;gap:14px}
.kv div:last-child{border-bottom:none}
.kv span{color:var(--ink3);flex:0 0 100px;font-weight:600}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px;padding-bottom:14px;border-bottom:1px solid var(--line2)}
.chip{background:var(--surf);border:1px solid var(--line2);border-radius:100px;padding:5px 13px;
font-size:12px;font-family:inherit;color:var(--ink2);cursor:pointer;display:flex;align-items:center;gap:6px}
.chip:hover{background:var(--surf1)}
.chip.active{background:var(--ink);color:#fff;border-color:var(--ink)}
.cm{font-size:11px;font-weight:600;line-height:1}
.cm.fixed{color:var(--green)}.cm.part{color:var(--amber)}.cm.open,.cm.none{color:var(--ink3)}
.cm.reg{color:var(--red)}.chip.active .cm{color:#fff}
.cd{background:var(--dark);color:#fff;font-size:9px;font-weight:600;padding:0 5px;border-radius:100px;line-height:1.5}
.cw{display:none}.cw.active{display:block}
.case{background:var(--surf);border:1px solid var(--line);border-radius:10px;padding:15px 17px}
.case-h{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:10px}
.ct{font-size:15px;font-weight:600}
.ci{font-size:12px;color:var(--ink2);font-style:italic;margin-top:2px}
.tag{font-size:11px;font-weight:600;padding:2px 9px;border-radius:100px;white-space:nowrap;flex:0 0 auto}
.t-fix{background:var(--greenbg);color:var(--green)}
.t-part{background:var(--amberbg);color:var(--amber)}
.t-open,.t-none{background:var(--surf0);color:var(--ink2);border:1px solid var(--line2)}
.t-reg{background:var(--redbg);color:var(--red)}
.why{background:var(--surf1);border-radius:7px;padding:9px 12px;font-size:12.5px;color:var(--ink2);margin-bottom:10px}
.wl{display:block;font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
color:var(--ink3);margin-bottom:3px}
.disc{background:var(--redbg);border:1px solid #F0997B;border-radius:7px;padding:9px 12px;margin-bottom:11px}
.dh{font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--red);margin-bottom:4px}
.dtag{background:var(--dark);color:#fff;font-size:9px;padding:1px 7px;border-radius:100px;
margin-left:6px;letter-spacing:0;text-transform:none}
.disc p{font-size:12.5px;color:var(--ink2)}
.subtabs{display:flex;gap:2px;background:var(--surf0);border-radius:7px;padding:3px;margin-bottom:11px;width:fit-content}
.st{background:none;border:none;font-family:inherit;font-size:12px;color:var(--ink2);
padding:5px 13px;border-radius:5px;cursor:pointer}
.st.active{background:var(--surf);color:var(--ink);font-weight:600;border:1px solid var(--line)}
.sv{display:none}.sv.active{display:block}
.two{display:grid;grid-template-columns:1fr 1fr;gap:13px}
.one{display:block}
.lab{font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--ink3);margin-bottom:5px}
.lab .n,.lab .prov{font-weight:400;letter-spacing:0;text-transform:none}
table.oc{width:100%;border-collapse:collapse;background:var(--surf1);border:1px solid var(--line);
border-radius:6px;overflow:hidden}
table.oc th{font-size:10px;font-weight:600;color:var(--ink2);text-align:left;padding:5px 8px;
background:var(--surf0);border-bottom:1px solid var(--line2)}
table.oc td{font-size:11px;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
table.oc tr:last-child td{border-bottom:none}
.oc-c{width:96px;font-weight:600;color:var(--ink2);white-space:nowrap}
.oc-t{color:var(--ink2);font-family:ui-monospace,Menlo,monospace;font-size:10.5px;line-height:1.45;word-break:break-word}
tr.hl-row{background:var(--redbg)}tr.hl-row .oc-c,tr.hl-row .oc-t{color:var(--red)}
.sv img{width:100%;height:auto;border:1px solid var(--line2);border-radius:6px;display:block;margin-top:5px}
.src{margin-top:11px;padding-top:9px;border-top:1px solid var(--line)}
.src a{font-size:12px;font-weight:600;color:var(--blue);text-decoration:none}
.src a:hover{text-decoration:underline}
.src .url{display:block;font-size:10px;color:var(--ink3);font-family:ui-monospace,Menlo,monospace;
word-break:break-all;margin-top:2px}
@media(max-width:900px){body{display:block}#nav{width:auto;height:auto;position:static;
border-right:none;border-bottom:1px solid var(--line2);display:flex;overflow-x:auto;padding:0}
.nav-i{border-left:none;border-bottom:3px solid transparent;white-space:nowrap;width:auto}
.nav-i.active{border-left-color:transparent;border-bottom-color:var(--blue)}
#main{padding:20px 14px 50px}.two{grid-template-columns:1fr}}
@media print{body{display:block}#nav,.chips,.subtabs{display:none}
.pane,.cw,.sv{display:block!important}.case{break-inside:avoid;margin-bottom:12px}}
</style></head><body>
"""

JS = """<script>
document.querySelectorAll(".nav-i").forEach(function(b){b.onclick=function(){
 document.querySelectorAll(".nav-i").forEach(function(x){x.classList.remove("active")});
 document.querySelectorAll(".pane").forEach(function(p){p.classList.remove("active")});
 b.classList.add("active");
 var e=document.getElementById("p-"+b.dataset.t); if(e)e.classList.add("active");
 window.scrollTo(0,0);};});
document.querySelectorAll(".chip").forEach(function(b){b.onclick=function(){
 var c=b.dataset.cat;
 document.querySelectorAll('.chip[data-cat="'+c+'"]').forEach(function(x){x.classList.remove("active")});
 document.querySelectorAll('[id^="w-'+c+'-"]').forEach(function(w){w.classList.remove("active")});
 b.classList.add("active");
 var e=document.getElementById("w-"+c+"-"+b.dataset.c); if(e)e.classList.add("active");};});
document.querySelectorAll(".st").forEach(function(b){b.onclick=function(){
 var c=b.dataset.c;
 b.parentElement.querySelectorAll(".st").forEach(function(x){x.classList.remove("active")});
 ["cmp","shot"].forEach(function(v){var e=document.getElementById("v-"+c+"-"+v);
   if(e)e.classList.remove("active")});
 b.classList.add("active");
 var e=document.getElementById("v-"+c+"-"+b.dataset.v); if(e)e.classList.add("active");};});
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["1.1", "1.2", "2.0"])
    ap.add_argument("--out-dir", default="./outputs")
    ap.add_argument("--reference", help="optional reference/verdict JSON (not committed)")
    ap.add_argument("--screenshots", help="optional base64 screenshot JSON (not committed)")
    ap.add_argument("--out", help="output path (default report/evidence-v<model>.html)")
    a = ap.parse_args()

    records = load_results(Path(a.out_dir), a.model)
    ref = load_json(a.reference, "Reference file")
    shots = load_json(a.screenshots, "Screenshot file")

    body = build(records, ref, shots, a.model)
    out = Path(a.out) if a.out else Path("report") / f"evidence-v{a.model}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(CSS + body + JS, encoding="utf-8")

    kb = out.stat().st_size // 1024
    print(f"{len(records)} pages -> {out}  ({kb} KB)")
    if not ref:
        print("No --reference supplied: rendered output only, no comparison or verdicts.")
    if not shots:
        print("No --screenshots supplied: reference screenshots omitted.")


if __name__ == "__main__":
    main()
