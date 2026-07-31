#!/usr/bin/env python3
"""
Nemotron-Parse bug-category evaluation.

Runs the 22 examples from the issue reports through a served Nemotron-Parse
model, scores each against expected element classes, records latency, and
writes per-model output folders.

Parsing uses the model repo's own postprocessing.py (extract_classes_bboxes),
NOT a hand-rolled regex — run --selftest first to confirm it works.

Usage
-----
  # 0. one-time: fetch the repo's postprocessing helper
  python3.10 nemotron_eval.py --fetch-postprocessing

  # 1. confirm parsing works against the running server (1 example)
  python3.10 nemotron_eval.py --selftest --model 2.0

  # 2. full run
  python3.10 nemotron_eval.py --model 2.0
  python3.10 nemotron_eval.py --model 1.2
  python3.10 nemotron_eval.py --model 1.1

  # 3. after all three, build the comparison table
  python3.10 nemotron_eval.py --compare

Options
-------
  --base-url   default http://localhost:8000/v1
  --out-dir    default ./outputs
  --limit N    only run the first N examples (smoke test)
"""

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

MODELS = {
    "2.0": {
        "id": "nvidia/NVIDIA-Nemotron-Parse-2.0",
        "prompt": "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>",
    },
    "1.2": {
        "id": "nvidia/NVIDIA-Nemotron-Parse-v1.2",
        "prompt": "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>",
    },
    "1.1": {
        # v1.1 predates the 4th prompt token
        "id": "nvidia/NVIDIA-Nemotron-Parse-v1.1",
        "prompt": "</s><s><predict_bbox><predict_classes><output_markdown>",
    },
}

# ---------------------------------------------------------------------------
# Eval set — 22 examples across the 7 bug categories from the issue PDFs
# expected = element classes that should be produced, in reading order
# ---------------------------------------------------------------------------

EXAMPLES = [
    # 1. Misclassification -------------------------------------------------
    dict(cat=1, cat_name="misclassification", eid="fao_p2",
         label="FAO document p.2",
         url="https://openknowledge.fao.org/server/api/core/bitstreams/c0ba979b-7122-456d-bc99-139dbf857bff/content",
         page=2, issue="Text tagged as Caption",
         expected=["Text", "Text", "Text", "Page-Footer"], known_errors=3),

    dict(cat=1, cat_name="misclassification", eid="q4_p13_misclass",
         label="Q4 Investor Presentation p.13",
         url="https://s203.q4cdn.com/249399152/files/doc_financials/2025/q4/Final-Q4-2025-Investor-Presentation.pdf",
         page=13, issue="Page-Footer marked as Caption",
         expected=["Section-header", "Text", "Page-Footer", "Text"], known_errors=1),

    dict(cat=1, cat_name="misclassification", eid="sclib_p2",
         label="SC State Library p.2",
         url="https://dc.statelibrary.sc.gov/server/api/core/bitstreams/8d086f8c-fb2b-4ae3-a022-0c123e105bc6/content",
         page=2, issue="Standalone text marked as Caption",
         expected=["Text", "Text", "Text", "Text"], known_errors=5),

    # 2. Figure detection --------------------------------------------------
    dict(cat=2, cat_name="figure_detection", eid="banana_p16",
         label="Banana specialty crop p.16",
         url="https://agroforestry.net/images/pdfs/Banana_specialty_crop.pdf",
         page=16, issue="Figure content missing",
         expected=["Picture", "Picture", "Caption", "Page-Footer"], known_errors=3),

    dict(cat=2, cat_name="figure_detection", eid="q4_p13_figure",
         label="Q4 Investor Presentation p.13 (graph)",
         url="https://s203.q4cdn.com/249399152/files/doc_financials/2025/q4/Final-Q4-2025-Investor-Presentation.pdf",
         page=13, issue="Graph split into text fragments",
         expected=["Picture", "Caption"], known_errors=6),

    # 3. Reading order -----------------------------------------------------
    dict(cat=3, cat_name="reading_order", eid="banana_p2",
         label="Banana specialty crop p.2",
         url="https://agroforestry.net/images/pdfs/Banana_specialty_crop.pdf",
         page=2, issue="Page-Footer before Picture/Caption",
         expected=["Section-header", "Text", "Picture", "Caption", "Page-Footer"], known_errors=2),

    dict(cat=3, cat_name="reading_order", eid="hamilton_p10",
         label="Hamilton Project p.10",
         url="https://www.hamiltonproject.org/wp-content/uploads/2023/01/twelve_economic_facts_energy_climate_change.pdf",
         page=10, issue="Caption at end instead of after figure",
         expected=["Footnote", "Page-Footer", "Picture", "Caption", "Picture", "Caption"], known_errors=2),

    dict(cat=3, cat_name="reading_order", eid="radisson_p6",
         label="Radisson Blu menu p.6",
         url="https://radissonhotels.iceportal.com/asset/radisson-blu-mbd-hotel-noida/miscellaneous/16256-114063-m26028486.pdf",
         page=6, issue="Prices not paired with dishes",
         expected=["Text", "Text", "Text", "Text", "Text"], known_errors=4),

    dict(cat=3, cat_name="reading_order", eid="sclib_p1",
         label="SC State Library p.1",
         url="https://dc.statelibrary.sc.gov/server/api/core/bitstreams/8d086f8c-fb2b-4ae3-a022-0c123e105bc6/content",
         page=1, issue="Captions grouped at end of page",
         expected=["Picture", "Caption", "Picture", "Caption", "Picture", "Caption"], known_errors=3),

    # 4. Form fields -------------------------------------------------------
    dict(cat=4, cat_name="form_fields", eid="toefl_p2",
         label="TOEFL registration form p.2",
         url="https://ankastudy.com/wp-content/uploads/toefl-register-form.pdf",
         page=2, issue="Form fields as Picture/Table",
         expected=["Text", "Text", "Text"], known_errors=4),

    dict(cat=4, cat_name="form_fields", eid="ucanr_p2",
         label="UC ANR appraisal form p.2",
         url="https://ucanr.edu/sites/default/files/2016-06/238147.pdf",
         page=2, issue="Checkbox arrays as Table/Picture",
         expected=["Text", "Text", "Text", "Text"], known_errors=5),

    dict(cat=4, cat_name="form_fields", eid="hyatt_p1",
         label="Hyatt CC auth form p.1",
         url="https://world.hyatt.com/content/dam/HyattStories/ccauth_hotels.pdf",
         page=1, issue="Form fields not detected",
         expected=["Title", "Text", "Text", "Text", "Text"], known_errors=6),

    dict(cat=4, cat_name="form_fields", eid="fgi_p1",
         label="Application form 24.27 FGI p.1",
         url="https://ebnds.com/wp-content/uploads/2024/09/Application-form-24.27-FGI.pdf",
         page=1, issue="Form body as Picture; address as Caption",
         expected=["Page-Footer", "Text", "Text", "Text"], known_errors=5),

    # 5. Header / footer ---------------------------------------------------
    dict(cat=5, cat_name="header_footer", eid="irs_p10",
         label="IRS 1040-ES p.10",
         url="https://www.irs.gov/pub/irs-prior/f1040es--2019.pdf",
         page=10, issue="Header/footer not tagged as artifacts",
         expected=["Page-Header", "Table", "Page-Footer", "Page-Footer"], known_errors=3),

    dict(cat=5, cat_name="header_footer", eid="seed_p4_header",
         label="2026 Seed Guide p.4 (header)",
         url="https://www.therightseed.com/content/dam/dpagco/therightseed/files/2026%20Seed%20Guide%20for%20Web.pdf",
         page=4, issue="Header image tagged as Picture",
         expected=["Page-Header", "Table", "Text", "Picture", "Page-Footer"], known_errors=2),

    dict(cat=5, cat_name="header_footer", eid="kalamazoo_p1",
         label="Kalamazoo permit form p.1",
         url="https://www.kalamazoocity.org/files/assets/public/v/5/applications-amp-forms/building-permits/tech-code-permit-2026.pdf",
         page=1, issue="Letterhead as Picture; footer misclassified",
         expected=["Page-Header", "Title", "Text", "Page-Footer"], known_errors=4),

    # 6. Complex tables ----------------------------------------------------
    dict(cat=6, cat_name="complex_tables", eid="seed_p4_table",
         label="2026 Seed Guide p.4 (table)",
         url="https://www.therightseed.com/content/dam/dpagco/therightseed/files/2026%20Seed%20Guide%20for%20Web.pdf",
         page=4, issue="In-cell images lost",
         expected=["Table", "Picture", "Picture", "Picture", "Picture"], known_errors=4),

    dict(cat=6, cat_name="complex_tables", eid="africa_p57",
         label="Economic Survey 2015 p.57",
         url="https://africacheck.org/sites/default/files/Economic-Survey-2015.pdf",
         page=57, issue="2nd table not detected",
         expected=["Table", "Table"], known_errors=1),

    dict(cat=6, cat_name="complex_tables", eid="africa_p67",
         label="Economic Survey 2015 p.67",
         url="https://africacheck.org/sites/default/files/Economic-Survey-2015.pdf",
         page=67, issue="Table internal order wrong",
         expected=["Table", "Caption", "Page-Footer"], known_errors=3),

    # 7. PPT / landscape ---------------------------------------------------
    dict(cat=7, cat_name="ppt_landscape", eid="q4_p12",
         label="Q4 Investor Presentation p.12",
         url="https://s203.q4cdn.com/249399152/files/doc_financials/2025/q4/Final-Q4-2025-Investor-Presentation.pdf",
         page=12, issue="Stat values as Picture; labels missing",
         expected=["Page-Header", "Title", "Page-Footer", "Text", "Text", "Text", "Text", "Text"],
         known_errors=5),

    dict(cat=7, cat_name="ppt_landscape", eid="q4_p6",
         label="Q4 Investor Presentation p.6",
         url="https://s203.q4cdn.com/249399152/files/doc_financials/2025/q4/Final-Q4-2025-Investor-Presentation.pdf",
         page=6, issue="Only header/footer tagged; body lost",
         expected=["Section-header", "Picture", "Text", "Text", "Text", "Text", "Text", "Page-Footer"],
         known_errors=20),

    dict(cat=7, cat_name="ppt_landscape", eid="firstmining_p3",
         label="First Mining Gold corporate deck p.3",
         url="https://firstmininggold.com/_resources/presentations/corporate-presentation.pdf",
         page=3, issue="Landscape slide: misclassification + loss",
         expected=["Title", "Text", "Picture", "Text", "Picture"], known_errors=6),
]

CAT_NAMES = {
    1: "Misclassification",
    2: "Figure detection",
    3: "Reading order",
    4: "Form fields",
    5: "Header/Footer",
    6: "Complex tables",
    7: "PPT / landscape",
}

# ---------------------------------------------------------------------------
# Postprocessing — uses the model repo's own helper
# ---------------------------------------------------------------------------

_extract = None


def load_extractor():
    """Import extract_classes_bboxes from the repo's postprocessing.py."""
    global _extract
    if _extract is not None:
        return _extract

    if not Path("postprocessing.py").exists():
        sys.exit(
            "ERROR: postprocessing.py not in cwd.\n"
            "Run:  python3.10 nemotron_eval.py --fetch-postprocessing"
        )

    try:
        from postprocessing import extract_classes_bboxes
    except Exception as e:
        import traceback
        print("ERROR: postprocessing.py exists but failed to import.")
        print(f"  {type(e).__name__}: {e}\n")
        traceback.print_exc()
        print("\nIf this is a missing module, install it and retry.")
        sys.exit(1)

    _extract = extract_classes_bboxes
    return _extract


def fetch_postprocessing(model_key="2.0"):
    """Download the repo's top-level .py helpers into cwd.

    postprocessing.py imports latex2html.py from the same repo, so fetching
    just the one file is not enough.
    """
    from huggingface_hub import snapshot_download
    import glob
    import os
    import shutil

    root = snapshot_download(MODELS[model_key]["id"], allow_patterns="*.py")
    n = 0
    for f in glob.glob(os.path.join(root, "*.py")):
        shutil.copy(f, ".")
        print(f"  copied {os.path.basename(f)}")
        n += 1
    print(f"\n{n} files from {MODELS[model_key]['id']}")

    try:
        from postprocessing import extract_classes_bboxes  # noqa: F401
        print("postprocessing imports cleanly.")
    except Exception as e:
        print(f"WARNING: postprocessing still fails to import: "
              f"{type(e).__name__}: {e}")


def parse_output(raw, img_w=None, img_h=None):
    """Return [{cls, text, bbox, bbox_px}] using the repo's extractor.

    bbox    - normalized 0..1 as the model emits
    bbox_px - pixel coords on the rendered page (needs img_w/img_h)
    """
    extract = load_extractor()
    try:
        classes, bboxes, texts = extract(raw)
    except Exception as e:
        print(f"    WARN: parse failed: {e}")
        return []

    to_orig = None
    if img_w and img_h:
        try:
            from postprocessing import transform_bbox_to_original
            to_orig = transform_bbox_to_original
        except ImportError:
            pass

    out = []
    for c, b, t in zip(classes, bboxes, texts):
        norm_bb = list(b) if b is not None else None
        px = None
        if norm_bb and img_w and img_h:
            if to_orig is not None:
                try:
                    px = [int(v) for v in to_orig(norm_bb, img_w, img_h)]
                except Exception:
                    px = None
            if px is None:  # fallback: plain scaling
                px = [int(norm_bb[0] * img_w), int(norm_bb[1] * img_h),
                      int(norm_bb[2] * img_w), int(norm_bb[3] * img_h)]
        out.append({"cls": c, "text": t, "bbox": norm_bb, "bbox_px": px})
    return out


# ---------------------------------------------------------------------------
# PDF page rendering
# ---------------------------------------------------------------------------

def load_input(ex, inputs_dir=Path("./inputs"), dpi=150,
               cache=Path("/tmp/nemotron_pdf_cache")):
    """Load the rendered page for an example.

    Prefers the pre-rendered PNG from prepare_inputs.py. Falls back to
    downloading only if that is missing — so sites that 403 on repeat
    requests do not break a run whose inputs were already prepared.
    """
    from PIL import Image

    local = inputs_dir / f"cat{ex['cat']}_{ex['cat_name']}" / f"{ex['eid']}.png"
    if local.exists():
        return Image.open(local).convert("RGB"), "local"

    import pypdfium2 as pdfium
    import requests

    cache.mkdir(parents=True, exist_ok=True)
    key = f"{abs(hash(ex['url'])) % (10 ** 10)}_p{ex['page']}.png"
    path = cache / key
    if path.exists():
        return Image.open(path).convert("RGB"), "cache"

    resp = requests.get(ex["url"], timeout=90,
                        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
    resp.raise_for_status()
    doc = pdfium.PdfDocument(resp.content)
    img = doc[ex["page"] - 1].render(scale=dpi / 72).to_pil()
    img.save(path)
    return img.convert("RGB"), "downloaded"


CLASS_COLORS = {
    "text": (46, 120, 214),
    "title": (153, 53, 86),
    "sectionheader": (153, 53, 86),
    "caption": (235, 104, 52),
    "picture": (59, 109, 17),
    "chart": (15, 110, 86),
    "table": (133, 79, 11),
    "pageheader": (83, 74, 183),
    "pagefooter": (83, 74, 183),
    "footnote": (120, 120, 120),
    "listitem": (0, 140, 160),
}


def draw_overlay(img, elements, dest):
    """Draw predicted boxes on the page, coloured by class."""
    from PIL import ImageDraw

    out = img.copy()
    d = ImageDraw.Draw(out, "RGBA")

    for i, el in enumerate(elements):
        bb = el.get("bbox_px")
        if not bb:
            continue
        x0, y0, x1, y1 = bb
        colour = CLASS_COLORS.get(norm(el["cls"]), (120, 120, 120))
        d.rectangle([x0, y0, x1, y1], outline=colour + (255,), width=3)
        tag = f"{i}:{el['cls']}"
        tw = 7 * len(tag) + 8
        d.rectangle([x0, max(0, y0 - 17), x0 + tw, y0], fill=colour + (230,))
        d.text((x0 + 4, max(0, y0 - 15)), tag, fill=(255, 255, 255))

    dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(dest)
    return dest


def to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer(client, model_id, prompt, img_b64, max_tokens=8192):
    """Single request. Returns (text, latency_dict)."""
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        }],
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body={
            "repetition_penalty": 1.1,
            "top_k": 1,
            "skip_special_tokens": False,   # required — classes are special tokens
        },
    )
    total = round(time.perf_counter() - t0, 3)
    usage = getattr(resp, "usage", None)
    toks = getattr(usage, "completion_tokens", None)
    return resp.choices[0].message.content, {
        "total_s": total,
        "tokens_out": toks,
        "tokens_per_sec": round(toks / total, 1) if toks and total else None,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def norm(c):
    return str(c).strip().lower().replace("-", "").replace("_", "").replace(" ", "")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def selftest(ver, base_url, out_root=None):
    from openai import OpenAI

    cfg = MODELS[ver]
    ex = EXAMPLES[3]  # banana p16 — figure-heavy, good parser exercise

    print(f"Self-test: {cfg['id']}")
    print(f"Example:   {ex['label']}\n")

    img, origin = load_input(ex)
    print(f"Input: {origin}  {img.width}x{img.height}")

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    raw, lat = infer(client, cfg["id"], cfg["prompt"], to_b64(img))

    print(f"Latency: {lat['total_s']}s  tokens={lat['tokens_out']}  "
          f"tps={lat['tokens_per_sec']}")
    print(f"Raw length: {len(raw)} chars\n")
    print("--- RAW (first 600 chars) " + "-" * 40)
    print(raw[:600])
    print("-" * 66 + "\n")

    parsed = parse_output(raw, img.width, img.height)
    print(f"PARSED: {len(parsed)} elements")
    for i, p in enumerate(parsed[:15]):
        txt = (p["text"] or "")[:44].replace("\n", " ")
        bb = p.get("bbox_px")
        bbs = f"[{bb[0]},{bb[1]},{bb[2]},{bb[3]}]" if bb else "no bbox"
        print(f"  {i:>2} {p['cls']:<15} {bbs:<24} {txt}")

    print()
    if parsed:
        print("Parser OK — safe to run the full eval.")
    else:
        print("Parser returned NOTHING. Do not run the full eval yet —")
        print("paste the RAW block above so the parser can be matched to it.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(ver, base_url, out_root, limit=None):
    from openai import OpenAI

    cfg = MODELS[ver]
    examples = EXAMPLES[:limit] if limit else EXAMPLES

    print("=" * 66)
    print(f"  Nemotron-Parse {ver}")
    print(f"  model:    {cfg['id']}")
    print(f"  endpoint: {base_url}")
    print(f"  examples: {len(examples)}")
    print("=" * 66 + "\n")

    load_extractor()
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    results = []

    for i, ex in enumerate(examples, 1):
        print(f"[{i:02d}/{len(examples)}] {ex['label']}  (p.{ex['page']})")

        try:
            img, origin = load_input(ex)
        except Exception as e:
            print(f"    SKIP - input unavailable: {e}\n")
            continue

        try:
            raw, lat = infer(client, cfg["id"], cfg["prompt"], to_b64(img))
        except Exception as e:
            print(f"    SKIP - inference failed: {e}\n")
            continue

        parsed = parse_output(raw, img.width, img.height)

        counts = {}
        for el in parsed:
            counts[el["cls"]] = counts.get(el["cls"], 0) + 1

        print(f"    input: {origin}  {img.width}x{img.height}")
        print(f"    {lat['total_s']}s  {lat['tokens_out']} tok  "
              f"{lat['tokens_per_sec']} tok/s")
        print(f"    {len(parsed)} elements: "
              + ", ".join(f"{k}×{v}" for k, v in sorted(counts.items())))

        d = out_root / f"v{ver}" / f"cat{ex['cat']}_{ex['cat_name']}"
        d.mkdir(parents=True, exist_ok=True)

        (d / f"{ex['eid']}_raw.txt").write_text(raw, encoding="utf-8")

        overlay_path = None
        if any(e.get("bbox_px") for e in parsed):
            try:
                overlay_path = draw_overlay(
                    img, parsed, d / f"{ex['eid']}_overlay.png")
                print(f"    overlay: {overlay_path.name}")
            except Exception as e:
                print(f"    WARN: overlay failed: {e}")

        rec = {
            "model": cfg["id"],
            "version": ver,
            "example": {k: ex[k] for k in ("cat", "cat_name", "eid", "label",
                                           "url", "page", "issue",
                                           "known_errors")},
            "image": {"width": img.width, "height": img.height,
                      "source": origin},
            "latency": lat,
            "n_elements": len(parsed),
            "class_counts": counts,
            "elements": parsed,
            "overlay": overlay_path.name if overlay_path else None,
        }
        (d / f"{ex['eid']}.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        results.append(rec)
        print()

    if not results:
        print("No examples completed.")
        return

    vd = out_root / f"v{ver}"
    vd.mkdir(parents=True, exist_ok=True)
    (vd / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    lats = [r["latency"]["total_s"] for r in results if r["latency"].get("total_s")]
    tps = [r["latency"]["tokens_per_sec"] for r in results
           if r["latency"].get("tokens_per_sec")]
    sl = sorted(lats)

    L = ["=" * 70,
         f"  Nemotron-Parse {ver} — run summary",
         "=" * 70, "",
         f"  examples completed : {len(results)}/{len(examples)}",
         f"  elements found     : {sum(r['n_elements'] for r in results)}", ""]
    if lats:
        L += [f"  latency mean : {sum(lats)/len(lats):.2f}s",
              f"          p50  : {sl[len(sl)//2]:.2f}s",
              f"          p90  : {sl[int(len(sl)*0.9)]:.2f}s",
              f"          max  : {sl[-1]:.2f}s"]
    if tps:
        L += [f"  throughput   : {sum(tps)/len(tps):.0f} tok/s mean"]
    L += ["", "  Per-example element counts:", "-" * 70]
    for r in results:
        cc = ", ".join(f"{k}×{v}" for k, v in sorted(r["class_counts"].items()))
        L.append(f"  {r['example']['eid']:<20} {r['n_elements']:>3}  {cc}")
    L += ["",
          "  Accuracy is NOT scored here. Run:  python3.10 score_bugs.py",
          "=" * 70]

    txt = "\n".join(L)
    (vd / "run_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\nSaved -> {vd}/results.json")
    print(f"Saved -> {vd}/run_summary.txt")
    print(f"Overlays in {vd}/cat*/")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["1.1", "1.2", "2.0"])
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--out-dir", default="./outputs")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--selftest", action="store_true",
                    help="run one example, print raw + parsed output")
    ap.add_argument("--fetch-postprocessing", action="store_true",
                    help="download the repo's .py helpers")
    a = ap.parse_args()

    out_root = Path(a.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if a.fetch_postprocessing:
        fetch_postprocessing()
    elif a.selftest:
        if not a.model:
            ap.error("--selftest requires --model")
        selftest(a.model, a.base_url, out_root)
    elif a.model:
        run(a.model, a.base_url, out_root, a.limit)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
