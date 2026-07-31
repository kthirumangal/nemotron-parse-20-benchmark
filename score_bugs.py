#!/usr/bin/env python3
"""
Nemotron-Parse — bug-signature scoring.

Instead of guessing full-page ground truth, this tests the SPECIFIC claim each
issue-report entry makes. Every check answers one question with PASS / FAIL /
N/A, and each is derived directly from the "Issue:" line in your PDFs.

Why not positional accuracy: a page has 6-40 elements and the report screenshots
only show the first few, so any hand-written expected list is wrong. Worse,
positional alignment turns one extra element at index 0 into a total miss — the
Economic Survey p.57 run found BOTH tables (the reported bug, fixed) and scored
0%.

Runs entirely on saved output. No GPU, no server.

Usage
-----
  python3.10 score_bugs.py                      # score everything present
  python3.10 score_bugs.py --model 2.0          # one model
  python3.10 score_bugs.py --detail             # show reasoning per check
  python3.10 score_bugs.py --csv results.csv    # export
"""

import argparse
import csv
import json
from pathlib import Path

# ---------------------------------------------------------------------------


def norm(c):
    return str(c).strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def classes(els):
    return [norm(e["cls"]) for e in els]


def y_center(el):
    bb = el.get("bbox") or el.get("bbox_px")
    if not bb:
        return None
    return (bb[1] + bb[3]) / 2


def overlaps_vertically(a, b, tol=0.02):
    """Do two elements share vertical space (normalized coords)?"""
    ba, bb = a.get("bbox"), b.get("bbox")
    if not ba or not bb:
        return False
    return not (ba[3] < bb[1] - tol or bb[3] < ba[1] - tol)


def nearest_picture_distance(el, els):
    """Normalized vertical gap from el to the closest Picture/Chart."""
    yc = y_center(el)
    if yc is None:
        return None
    best = None
    for o in els:
        if norm(o["cls"]) not in ("picture", "chart", "figure"):
            continue
        yo = y_center(o)
        if yo is None:
            continue
        d = abs(yc - yo)
        best = d if best is None else min(best, d)
    return best


# ---------------------------------------------------------------------------
# Checks — one function per bug signature.
# Each returns (verdict, detail) where verdict is "PASS" | "FAIL" | "N/A".
# PASS means the reported bug is NOT present (i.e. behaviour is correct).
# ---------------------------------------------------------------------------

def check_orphan_captions(els, **kw):
    """Bug: Caption used for text that has no associated image.

    A Caption far from any Picture/Chart is an orphan.
    """
    caps = [e for e in els if norm(e["cls"]) == "caption"]
    if not caps:
        return "PASS", "no Caption elements emitted"

    pics = [e for e in els if norm(e["cls"]) in ("picture", "chart", "figure")]
    if not pics:
        return "FAIL", f"{len(caps)} Caption(s) but zero Picture/Chart on page"

    orphans = []
    for c in caps:
        d = nearest_picture_distance(c, els)
        if d is None or d > 0.25:
            orphans.append(c)

    if orphans:
        return "FAIL", (f"{len(orphans)}/{len(caps)} Caption(s) >0.25 page-height "
                        f"from any figure")
    return "PASS", f"all {len(caps)} Caption(s) near a figure"


def check_footer_is_last(els, **kw):
    """Bug: Page-Footer emitted mid-sequence instead of at the end."""
    idx = [i for i, e in enumerate(els) if norm(e["cls"]) == "pagefooter"]
    if not idx:
        return "N/A", "no Page-Footer emitted"

    last_footer = max(idx)
    after = len(els) - 1 - last_footer
    if after == 0:
        return "PASS", "Page-Footer is final element"

    # tolerate trailing footnotes
    trailing = [norm(els[i]["cls"]) for i in range(last_footer + 1, len(els))]
    if all(t in ("footnote", "pagefooter") for t in trailing):
        return "PASS", f"only {trailing} follow the footer"
    return "FAIL", f"{after} element(s) after Page-Footer: {trailing[:5]}"


def check_header_is_first(els, **kw):
    """Bug: Page-Header not at the start of the sequence."""
    idx = [i for i, e in enumerate(els) if norm(e["cls"]) == "pageheader"]
    if not idx:
        return "N/A", "no Page-Header emitted"
    if min(idx) <= 1:
        return "PASS", f"Page-Header at index {min(idx)}"
    return "FAIL", f"Page-Header at index {min(idx)}, expected 0 or 1"


def check_header_footer_present(els, **kw):
    """Bug: header/footer artifacts tagged as Text or Picture instead."""
    cls = classes(els)
    has_h = "pageheader" in cls
    has_f = "pagefooter" in cls
    if has_h and has_f:
        return "PASS", "both Page-Header and Page-Footer tagged"
    if has_h or has_f:
        return "FAIL", f"only {'header' if has_h else 'footer'} tagged"
    return "FAIL", "neither Page-Header nor Page-Footer tagged"


def check_caption_follows_figure(els, **kw):
    """Bug: Captions grouped at end of page rather than after each figure."""
    caps = [(i, e) for i, e in enumerate(els) if norm(e["cls"]) == "caption"]
    if len(caps) < 2:
        return "N/A", f"{len(caps)} Caption(s) — need 2+ to judge grouping"

    n = len(els)
    positions = [i for i, _ in caps]
    # all captions in the final third => grouped at end
    if all(p > n * 0.66 for p in positions):
        return "FAIL", f"all {len(caps)} Captions in final third (idx {positions})"

    # check each caption follows a figure reasonably closely
    bad = 0
    for i, c in caps:
        prev = [norm(e["cls"]) for e in els[max(0, i - 3):i]]
        if not any(p in ("picture", "chart", "figure", "table") for p in prev):
            bad += 1
    if bad > len(caps) / 2:
        return "FAIL", f"{bad}/{len(caps)} Captions not preceded by a figure"
    return "PASS", f"Captions interleaved (idx {positions})"


def check_figure_detected(els, **kw):
    """Bug: figure fragmented into text; not detected as a unit."""
    cls = classes(els)
    n_fig = sum(1 for c in cls if c in ("picture", "chart", "figure"))
    if n_fig == 0:
        return "FAIL", "no Picture/Chart/Figure detected on a figure-bearing page"
    return "PASS", f"{n_fig} figure element(s) detected"


def check_chart_class(els, **kw):
    """2.0-only: is the Chart class being used at all?"""
    n = sum(1 for c in classes(els) if c == "chart")
    if n:
        return "PASS", f"{n} Chart element(s)"
    return "N/A", "no Chart emitted (expected for v1.1/v1.2)"


def check_table_count(els, expect_min=1, **kw):
    """Bug: table on page not detected (e.g. '2nd table not tagged')."""
    n = sum(1 for c in classes(els) if c == "table")
    if n >= expect_min:
        return "PASS", f"{n} Table(s) detected, expected >={expect_min}"
    return "FAIL", f"only {n} Table(s), expected >={expect_min}"


def check_form_not_image(els, **kw):
    """Bug: form fields swallowed into a Picture or a single Table."""
    cls = classes(els)
    n_text = sum(1 for c in cls if c in ("text", "listitem", "title",
                                         "sectionheader"))
    n_pic = sum(1 for c in cls if c in ("picture", "chart", "figure"))

    if n_text == 0:
        return "FAIL", f"no text elements — form absorbed ({n_pic} Picture)"
    if n_text < 3 and n_pic > 0:
        return "FAIL", f"only {n_text} text vs {n_pic} Picture — likely absorbed"
    return "PASS", f"{n_text} text element(s) extracted from form"


def check_content_not_lost(els, min_elements=5, **kw):
    """Bug: slide body content missing; only header/footer tagged."""
    cls = classes(els)
    body = [c for c in cls if c not in ("pageheader", "pagefooter")]
    if len(body) < min_elements:
        return "FAIL", (f"only {len(body)} body element(s) "
                        f"(expected >={min_elements}) — content loss")
    return "PASS", f"{len(body)} body elements"


def check_text_not_picture(els, **kw):
    """Bug: text values (stats, labels) tagged as Picture."""
    cls = classes(els)
    n_text = sum(1 for c in cls if c in ("text", "listitem"))
    n_pic = sum(1 for c in cls if c in ("picture", "chart", "figure"))
    if n_pic > 0 and n_text == 0:
        return "FAIL", f"{n_pic} Picture, 0 Text — text likely absorbed"
    if n_pic > 3 * max(n_text, 1):
        return "FAIL", f"{n_pic} Picture vs {n_text} Text — skewed"
    return "PASS", f"{n_text} Text vs {n_pic} Picture"


# ---------------------------------------------------------------------------
# Which checks apply to which example
# ---------------------------------------------------------------------------

CHECKS = {
    "fao_p2": [
        ("no orphan captions", check_orphan_captions, {}),
        ("footer last", check_footer_is_last, {}),
    ],
    "q4_p13_misclass": [
        ("no orphan captions", check_orphan_captions, {}),
        ("footer tagged", check_header_footer_present, {}),
    ],
    "sclib_p2": [
        ("no orphan captions", check_orphan_captions, {}),
    ],
    "banana_p16": [
        ("figure detected", check_figure_detected, {}),
        ("chart class used", check_chart_class, {}),
    ],
    "q4_p13_figure": [
        ("figure detected", check_figure_detected, {}),
        ("chart class used", check_chart_class, {}),
    ],
    "banana_p2": [
        ("footer last", check_footer_is_last, {}),
        ("caption follows figure", check_caption_follows_figure, {}),
    ],
    "hamilton_p10": [
        ("caption follows figure", check_caption_follows_figure, {}),
        ("footer last", check_footer_is_last, {}),
    ],
    "radisson_p6": [
        ("no orphan captions", check_orphan_captions, {}),
        ("content not lost", check_content_not_lost, {"min_elements": 10}),
    ],
    "sclib_p1": [
        ("caption follows figure", check_caption_follows_figure, {}),
        ("figure detected", check_figure_detected, {}),
    ],
    "toefl_p2": [
        ("form not absorbed", check_form_not_image, {}),
    ],
    "ucanr_p2": [
        ("form not absorbed", check_form_not_image, {}),
        ("content not lost", check_content_not_lost, {"min_elements": 8}),
    ],
    "hyatt_p1": [
        ("form not absorbed", check_form_not_image, {}),
        ("content not lost", check_content_not_lost, {"min_elements": 8}),
    ],
    "fgi_p1": [
        ("form not absorbed", check_form_not_image, {}),
        ("no orphan captions", check_orphan_captions, {}),
    ],
    "irs_p10": [
        ("header+footer tagged", check_header_footer_present, {}),
        ("header first", check_header_is_first, {}),
        ("footer last", check_footer_is_last, {}),
    ],
    "seed_p4_header": [
        ("header+footer tagged", check_header_footer_present, {}),
        ("header first", check_header_is_first, {}),
    ],
    "kalamazoo_p1": [
        ("header+footer tagged", check_header_footer_present, {}),
        ("form not absorbed", check_form_not_image, {}),
    ],
    "seed_p4_table": [
        ("table detected", check_table_count, {"expect_min": 1}),
        ("content not lost", check_content_not_lost, {"min_elements": 8}),
    ],
    "africa_p57": [
        ("both tables found", check_table_count, {"expect_min": 2}),
        ("footer last", check_footer_is_last, {}),
    ],
    "africa_p67": [
        ("table detected", check_table_count, {"expect_min": 1}),
        ("footer last", check_footer_is_last, {}),
    ],
    "q4_p12": [
        ("content not lost", check_content_not_lost, {"min_elements": 5}),
        ("text not as picture", check_text_not_picture, {}),
    ],
    "q4_p6": [
        ("content not lost", check_content_not_lost, {"min_elements": 6}),
        ("text not as picture", check_text_not_picture, {}),
    ],
    "firstmining_p3": [
        ("content not lost", check_content_not_lost, {"min_elements": 6}),
        ("text not as picture", check_text_not_picture, {}),
    ],
}

CAT_NAMES = {
    1: "Misclassification", 2: "Figure detection", 3: "Reading order",
    4: "Form fields", 5: "Header/Footer", 6: "Complex tables",
    7: "PPT / landscape",
}

# ---------------------------------------------------------------------------


def score_model(ver, out_root, detail=False):
    vd = out_root / f"v{ver}"
    if not vd.exists():
        return None

    rows = []
    for jf in sorted(vd.glob("cat*/*.json")):
        rec = json.loads(jf.read_text())
        ex = rec["example"]
        eid = ex["eid"]
        els = rec.get("elements", [])

        for name, fn, kw in CHECKS.get(eid, []):
            try:
                verdict, why = fn(els, **kw)
            except Exception as e:
                verdict, why = "N/A", f"check error: {e}"
            rows.append({
                "model": ver,
                "cat": ex["cat"],
                "category": CAT_NAMES[ex["cat"]],
                "eid": eid,
                "label": ex["label"],
                "issue": ex["issue"],
                "check": name,
                "verdict": verdict,
                "detail": why,
                "n_elements": rec.get("n_elements", len(els)),
                "latency_s": rec.get("latency", {}).get("total_s"),
                "tokens": rec.get("latency", {}).get("tokens_out"),
            })
    return rows


def print_report(rows, ver, detail=False):
    W = 92
    print("=" * W)
    print(f"  Nemotron-Parse {ver} — bug-signature results")
    print("=" * W)
    print()
    print("  PASS = reported bug NOT present    FAIL = bug reproduces")
    print()

    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["cat"], []).append(r)

    print(f"  {'Category':<22} {'Checks':>7} {'Pass':>6} {'Fail':>6} "
          f"{'N/A':>5} {'Pass %':>8}")
    print("  " + "-" * (W - 4))

    tp = tf = tn = 0
    for cid in sorted(by_cat):
        g = by_cat[cid]
        p = sum(1 for r in g if r["verdict"] == "PASS")
        f = sum(1 for r in g if r["verdict"] == "FAIL")
        n = sum(1 for r in g if r["verdict"] == "N/A")
        d = p + f
        pct = f"{p/d*100:.0f}%" if d else "—"
        print(f"  {CAT_NAMES[cid]:<22} {len(g):>7} {p:>6} {f:>6} {n:>5} {pct:>8}")
        tp += p; tf += f; tn += n

    print("  " + "-" * (W - 4))
    d = tp + tf
    pct = f"{tp/d*100:.0f}%" if d else "—"
    print(f"  {'TOTAL':<22} {len(rows):>7} {tp:>6} {tf:>6} {tn:>5} {pct:>8}")
    print()

    if detail:
        print("  Per-check detail")
        print("  " + "-" * (W - 4))
        cur = None
        for r in rows:
            if r["eid"] != cur:
                cur = r["eid"]
                print(f"\n  {r['label']}")
                print(f"    reported: {r['issue']}")
                print(f"    elements: {r['n_elements']}   "
                      f"latency: {r['latency_s']}s")
            mark = {"PASS": "PASS", "FAIL": "FAIL", "N/A": " n/a"}[r["verdict"]]
            print(f"    [{mark}] {r['check']:<24} {r['detail']}")
        print()


def print_comparison(all_rows):
    versions = sorted({r["model"] for r in all_rows})
    if len(versions) < 2:
        return

    W = 92
    print("=" * W)
    print("  Cross-model — bug signatures resolved")
    print("=" * W)
    print()

    hdr = f"  {'Category':<22}"
    for v in versions:
        hdr += f"{v:>12}"
    print(hdr)
    print("  " + "-" * (W - 4))

    for cid in sorted(CAT_NAMES):
        line = f"  {CAT_NAMES[cid]:<22}"
        for v in versions:
            g = [r for r in all_rows if r["model"] == v and r["cat"] == cid]
            p = sum(1 for r in g if r["verdict"] == "PASS")
            f = sum(1 for r in g if r["verdict"] == "FAIL")
            line += f"{(f'{p}/{p+f}' if p + f else '—'):>12}"
        print(line)

    print("  " + "-" * (W - 4))
    line = f"  {'TOTAL':<22}"
    for v in versions:
        g = [r for r in all_rows if r["model"] == v]
        p = sum(1 for r in g if r["verdict"] == "PASS")
        f = sum(1 for r in g if r["verdict"] == "FAIL")
        line += f"{(f'{p}/{p+f}' if p + f else '—'):>12}"
    print(line)
    print()

    # latency
    print(f"  {'Latency (mean)':<22}", end="")
    for v in versions:
        lats = [r["latency_s"] for r in all_rows
                if r["model"] == v and r["latency_s"]]
        seen, uniq = set(), []
        for r in all_rows:
            if r["model"] == v and r["eid"] not in seen and r["latency_s"]:
                seen.add(r["eid"])
                uniq.append(r["latency_s"])
        print(f"{(f'{sum(uniq)/len(uniq):.2f}s' if uniq else '—'):>12}", end="")
    print("\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="./outputs")
    ap.add_argument("--model", choices=["1.1", "1.2", "2.0"])
    ap.add_argument("--detail", action="store_true",
                    help="print reasoning for every check")
    ap.add_argument("--csv", help="write all rows to CSV")
    a = ap.parse_args()

    out_root = Path(a.out_dir)
    versions = [a.model] if a.model else ["2.0", "1.2", "1.1"]

    all_rows = []
    for v in versions:
        rows = score_model(v, out_root, a.detail)
        if not rows:
            if a.model:
                print(f"No results found in {out_root / ('v' + v)}")
            continue
        print_report(rows, v, a.detail)
        (out_root / f"v{v}" / "bug_signatures.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        all_rows += rows

    if not all_rows:
        print("Nothing to score. Run nemotron_eval.py first.")
        return

    print_comparison(all_rows)

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"CSV -> {a.csv}")

    print(f"Per-model JSON -> {out_root}/v*/bug_signatures.json")


if __name__ == "__main__":
    main()
