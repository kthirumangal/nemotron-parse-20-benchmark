#!/usr/bin/env python3
"""
Nemotron-Parse eval — input preprocessing.

Downloads every source PDF referenced in the issue reports, renders the
specific page under test, and writes it to an inputs/ folder alongside a
manifest tying each image back to its source, expected classes, and the
issue it demonstrates.

Run this BEFORE benchmarking. It surfaces dead URLs and pagination drift up
front — during a benchmark run those become silent SKIPs that shrink the
denominator and quietly flatter the model.

Usage
-----
  python3.10 prepare_inputs.py                    # render all
  python3.10 prepare_inputs.py --dpi 200          # higher resolution
  python3.10 prepare_inputs.py --cat 4            # one category only
  python3.10 prepare_inputs.py --force            # ignore cache, re-download
  python3.10 prepare_inputs.py --check            # verify existing, no download
  python3.10 prepare_inputs.py --contact-sheet    # build a review montage

Output
------
  inputs/                    <- fixtures, sibling of outputs/ not inside it
    cat1_misclassification/
      fao_p2.png
      ...
    manifest.json
    contact_sheet.png        (with --contact-sheet)
    FAILED.json              (only if something failed)

  inputs/ is deliberately NOT under outputs/. Outputs are per-model results
  you will delete and regenerate; inputs are verified fixtures that should
  survive that.

Requires: pypdfium2 pillow requests
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Eval set — must stay in sync with nemotron_eval.py EXAMPLES
# ---------------------------------------------------------------------------

EXAMPLES = [
    # 1. Misclassification -------------------------------------------------
    dict(cat=1, cat_name="misclassification", eid="fao_p2",
         label="FAO document p.2",
         url="https://openknowledge.fao.org/server/api/core/bitstreams/c0ba979b-7122-456d-bc99-139dbf857bff/content",
         page=2, issue="Text tagged as Caption",
         expected=["Text", "Text", "Text", "Page-Footer"], known_errors=3,
         source_doc="rev2"),

    dict(cat=1, cat_name="misclassification", eid="q4_p13_misclass",
         label="Q4 Investor Presentation p.13",
         url="https://s203.q4cdn.com/249399152/files/doc_financials/2025/q4/Final-Q4-2025-Investor-Presentation.pdf",
         page=13, issue="Page-Footer marked as Caption",
         expected=["Section-header", "Text", "Page-Footer", "Text"], known_errors=1,
         source_doc="rev2"),

    dict(cat=1, cat_name="misclassification", eid="sclib_p2",
         label="SC State Library p.2",
         url="https://dc.statelibrary.sc.gov/server/api/core/bitstreams/8d086f8c-fb2b-4ae3-a022-0c123e105bc6/content",
         page=2, issue="Standalone text marked as Caption",
         expected=["Text", "Text", "Text", "Text"], known_errors=5,
         source_doc="issues_1"),

    # 2. Figure detection --------------------------------------------------
    dict(cat=2, cat_name="figure_detection", eid="banana_p16",
         label="Banana specialty crop p.16",
         url="https://agroforestry.net/images/pdfs/Banana_specialty_crop.pdf",
         page=16, issue="Figure content missing",
         expected=["Picture", "Picture", "Caption", "Page-Footer"], known_errors=3,
         source_doc="rev2"),

    dict(cat=2, cat_name="figure_detection", eid="q4_p13_figure",
         label="Q4 Investor Presentation p.13 (graph)",
         url="https://s203.q4cdn.com/249399152/files/doc_financials/2025/q4/Final-Q4-2025-Investor-Presentation.pdf",
         page=13, issue="Graph split into text fragments",
         expected=["Picture", "Caption"], known_errors=6,
         source_doc="rev2"),

    # 3. Reading order -----------------------------------------------------
    dict(cat=3, cat_name="reading_order", eid="banana_p2",
         label="Banana specialty crop p.2",
         url="https://agroforestry.net/images/pdfs/Banana_specialty_crop.pdf",
         page=2, issue="Page-Footer before Picture/Caption",
         expected=["Section-header", "Text", "Picture", "Caption", "Page-Footer"],
         known_errors=2, source_doc="rev2"),

    dict(cat=3, cat_name="reading_order", eid="hamilton_p10",
         label="Hamilton Project p.10",
         url="https://www.hamiltonproject.org/wp-content/uploads/2023/01/twelve_economic_facts_energy_climate_change.pdf",
         page=10, issue="Caption at end instead of after figure",
         expected=["Footnote", "Page-Footer", "Picture", "Caption", "Picture", "Caption"],
         known_errors=2, source_doc="rev2"),

    dict(cat=3, cat_name="reading_order", eid="radisson_p6",
         label="Radisson Blu menu p.6",
         url="https://radissonhotels.iceportal.com/asset/radisson-blu-mbd-hotel-noida/miscellaneous/16256-114063-m26028486.pdf",
         page=6, issue="Prices not paired with dishes",
         expected=["Text", "Text", "Text", "Text", "Text"], known_errors=4,
         source_doc="rev2"),

    dict(cat=3, cat_name="reading_order", eid="sclib_p1",
         label="SC State Library p.1",
         url="https://dc.statelibrary.sc.gov/server/api/core/bitstreams/8d086f8c-fb2b-4ae3-a022-0c123e105bc6/content",
         page=1, issue="Captions grouped at end of page",
         expected=["Picture", "Caption", "Picture", "Caption", "Picture", "Caption"],
         known_errors=3, source_doc="issues_1"),

    # 4. Form fields -------------------------------------------------------
    dict(cat=4, cat_name="form_fields", eid="toefl_p2",
         label="TOEFL registration form p.2",
         url="https://ankastudy.com/wp-content/uploads/toefl-register-form.pdf",
         page=2, issue="Form fields as Picture/Table",
         expected=["Text", "Text", "Text"], known_errors=4,
         source_doc="rev2"),

    dict(cat=4, cat_name="form_fields", eid="ucanr_p2",
         label="UC ANR appraisal form p.2",
         url="https://ucanr.edu/sites/default/files/2016-06/238147.pdf",
         page=2, issue="Checkbox arrays as Table/Picture",
         expected=["Text", "Text", "Text", "Text"], known_errors=5,
         source_doc="rev2"),

    dict(cat=4, cat_name="form_fields", eid="hyatt_p1",
         label="Hyatt CC auth form p.1",
         url="https://world.hyatt.com/content/dam/HyattStories/ccauth_hotels.pdf",
         page=1, issue="Form fields not detected",
         expected=["Title", "Text", "Text", "Text", "Text"], known_errors=6,
         source_doc="rev2"),

    dict(cat=4, cat_name="form_fields", eid="fgi_p1",
         label="Application form 24.27 FGI p.1",
         url="https://ebnds.com/wp-content/uploads/2024/09/Application-form-24.27-FGI.pdf",
         page=1, issue="Form body as Picture; address as Caption",
         expected=["Page-Footer", "Text", "Text", "Text"], known_errors=5,
         source_doc="issues_1"),

    # 5. Header / footer ---------------------------------------------------
    dict(cat=5, cat_name="header_footer", eid="irs_p10",
         label="IRS 1040-ES p.10",
         url="https://www.irs.gov/pub/irs-prior/f1040es--2019.pdf",
         page=10, issue="Header/footer not tagged as artifacts",
         expected=["Page-Header", "Table", "Page-Footer", "Page-Footer"], known_errors=3,
         source_doc="rev2"),

    dict(cat=5, cat_name="header_footer", eid="seed_p4_header",
         label="2026 Seed Guide p.4 (header)",
         url="https://www.therightseed.com/content/dam/dpagco/therightseed/files/2026%20Seed%20Guide%20for%20Web.pdf",
         page=4, issue="Header image tagged as Picture",
         expected=["Page-Header", "Table", "Text", "Picture", "Page-Footer"], known_errors=2,
         source_doc="rev2"),

    dict(cat=5, cat_name="header_footer", eid="kalamazoo_p1",
         label="Kalamazoo permit form p.1",
         url="https://www.kalamazoocity.org/files/assets/public/v/5/applications-amp-forms/building-permits/tech-code-permit-2026.pdf",
         page=1, issue="Letterhead as Picture; footer misclassified",
         expected=["Page-Header", "Title", "Text", "Page-Footer"], known_errors=4,
         source_doc="issues_1"),

    # 6. Complex tables ----------------------------------------------------
    dict(cat=6, cat_name="complex_tables", eid="seed_p4_table",
         label="2026 Seed Guide p.4 (table)",
         url="https://www.therightseed.com/content/dam/dpagco/therightseed/files/2026%20Seed%20Guide%20for%20Web.pdf",
         page=4, issue="In-cell images lost",
         expected=["Table", "Picture", "Picture", "Picture", "Picture"], known_errors=4,
         source_doc="rev2"),

    dict(cat=6, cat_name="complex_tables", eid="africa_p57",
         label="Economic Survey 2015 p.57",
         url="https://africacheck.org/sites/default/files/Economic-Survey-2015.pdf",
         page=57, issue="2nd table not detected",
         expected=["Table", "Table"], known_errors=1,
         source_doc="rev2"),

    dict(cat=6, cat_name="complex_tables", eid="africa_p67",
         label="Economic Survey 2015 p.67",
         url="https://africacheck.org/sites/default/files/Economic-Survey-2015.pdf",
         page=67, issue="Table internal order wrong",
         expected=["Table", "Caption", "Page-Footer"], known_errors=3,
         source_doc="rev2"),

    # 7. PPT / landscape ---------------------------------------------------
    dict(cat=7, cat_name="ppt_landscape", eid="q4_p12",
         label="Q4 Investor Presentation p.12",
         url="https://s203.q4cdn.com/249399152/files/doc_financials/2025/q4/Final-Q4-2025-Investor-Presentation.pdf",
         page=12, issue="Stat values as Picture; labels missing",
         expected=["Page-Header", "Title", "Page-Footer", "Text", "Text", "Text", "Text", "Text"],
         known_errors=5, source_doc="rev2"),

    dict(cat=7, cat_name="ppt_landscape", eid="q4_p6",
         label="Q4 Investor Presentation p.6",
         url="https://s203.q4cdn.com/249399152/files/doc_financials/2025/q4/Final-Q4-2025-Investor-Presentation.pdf",
         page=6, issue="Only header/footer tagged; body lost",
         expected=["Section-header", "Picture", "Text", "Text", "Text", "Text", "Text", "Page-Footer"],
         known_errors=20, source_doc="rev2"),

    dict(cat=7, cat_name="ppt_landscape", eid="firstmining_p3",
         label="First Mining Gold corporate deck p.3",
         url="https://firstmininggold.com/_resources/presentations/corporate-presentation.pdf",
         page=3, issue="Landscape slide: misclassification + loss",
         expected=["Title", "Text", "Picture", "Text", "Picture"], known_errors=6,
         source_doc="issues_1"),
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

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

# ---------------------------------------------------------------------------


def pdf_cache_path(url, cache_dir):
    """Stable filename for a downloaded PDF."""
    import hashlib
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    return cache_dir / f"{h}.pdf"


def download_pdf(url, cache_dir, force=False, timeout=120):
    """Download a PDF once and keep it. Returns (path, bytes_len, from_cache)."""
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = pdf_cache_path(url, cache_dir)

    if path.exists() and not force:
        return path, path.stat().st_size, True

    resp = requests.get(url, timeout=timeout, headers=HEADERS)
    resp.raise_for_status()

    body = resp.content
    if not body.startswith(b"%PDF"):
        ct = resp.headers.get("content-type", "?")
        raise ValueError(
            f"response is not a PDF (content-type={ct}, {len(body)} bytes) — "
            "URL may redirect to a login or landing page"
        )

    path.write_bytes(body)
    return path, len(body), False


def render(pdf_path, page, dpi):
    """Render one 1-indexed page to a PIL image. Returns (img, n_pages)."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    n = len(doc)
    if page < 1 or page > n:
        raise IndexError(f"page {page} out of range — document has {n} pages")
    img = doc[page - 1].render(scale=dpi / 72).to_pil()
    return img, n


def contact_sheet(inputs_dir, examples, cols=4, thumb=380):
    """Montage of every rendered input for quick visual review."""
    from PIL import Image, ImageDraw

    tiles = []
    for ex in examples:
        p = inputs_dir / f"cat{ex['cat']}_{ex['cat_name']}" / f"{ex['eid']}.png"
        if p.exists():
            tiles.append((ex, p))

    if not tiles:
        print("No rendered inputs to build a contact sheet from.")
        return

    label_h = 34
    cell_w, cell_h = thumb, thumb + label_h
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    for i, (ex, p) in enumerate(tiles):
        img = Image.open(p)
        img.thumbnail((thumb - 10, thumb - 10))
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        sheet.paste(img, (x + 5, y + label_h))
        draw.text((x + 5, y + 4), f"{ex['eid']}  (cat {ex['cat']})", fill="black")
        draw.text((x + 5, y + 18), ex["label"][:52], fill="#555555")
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline="#cccccc")

    out = inputs_dir / "contact_sheet.png"
    sheet.save(out)
    print(f"Contact sheet -> {out}  ({len(tiles)} tiles)")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs-dir", default="./inputs",
                    help="where rendered pages are written (default ./inputs)")
    ap.add_argument("--pdf-cache", default="/tmp/nemotron_pdfs",
                    help="where downloaded PDFs are kept")
    ap.add_argument("--dpi", type=int, default=150,
                    help="render resolution (default 150)")
    ap.add_argument("--cat", type=int, choices=range(1, 8),
                    help="only process one category")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if cached")
    ap.add_argument("--check", action="store_true",
                    help="verify already-rendered inputs, no network")
    ap.add_argument("--contact-sheet", action="store_true",
                    help="also build a review montage")
    a = ap.parse_args()

    inputs_dir = Path(a.inputs_dir)
    cache_dir = Path(a.pdf_cache)

    examples = [e for e in EXAMPLES if a.cat is None or e["cat"] == a.cat]

    # -- check mode ---------------------------------------------------------
    if a.check:
        print(f"Checking {inputs_dir}\n")
        missing = []
        for ex in examples:
            p = inputs_dir / f"cat{ex['cat']}_{ex['cat_name']}" / f"{ex['eid']}.png"
            if p.exists():
                from PIL import Image
                w, h = Image.open(p).size
                print(f"  OK      {ex['eid']:<22} {w}x{h}")
            else:
                print(f"  MISSING {ex['eid']:<22} {p}")
                missing.append(ex["eid"])
        print(f"\n{len(examples) - len(missing)}/{len(examples)} present")
        if missing:
            print("Run without --check to render the missing ones.")
            sys.exit(1)
        return

    # -- render -------------------------------------------------------------
    print("=" * 70)
    print(f"  Preparing {len(examples)} eval inputs at {a.dpi} DPI")
    print(f"  inputs -> {inputs_dir}")
    print(f"  pdfs   -> {cache_dir}")
    print("=" * 70 + "\n")

    inputs_dir.mkdir(parents=True, exist_ok=True)

    manifest, failed = [], []
    t_start = time.time()

    for i, ex in enumerate(examples, 1):
        tag = f"[{i:02d}/{len(examples)}]"
        print(f"{tag} {ex['label']}")
        print(f"       cat {ex['cat']} ({CAT_NAMES[ex['cat']]})  ·  page {ex['page']}")

        try:
            pdf_path, size, cached = download_pdf(ex["url"], cache_dir, a.force)
            print(f"       pdf: {size/1e6:.1f} MB {'(cached)' if cached else '(downloaded)'}")

            img, n_pages = render(pdf_path, ex["page"], a.dpi)

            d = inputs_dir / f"cat{ex['cat']}_{ex['cat_name']}"
            d.mkdir(parents=True, exist_ok=True)
            dest = d / f"{ex['eid']}.png"
            img.save(dest)

            orientation = "landscape" if img.width > img.height else "portrait"
            print(f"       page {ex['page']}/{n_pages}  {img.width}x{img.height}  {orientation}")
            print(f"       -> {dest.relative_to(inputs_dir)}")

            manifest.append({
                "eid": ex["eid"],
                "cat": ex["cat"],
                "cat_name": ex["cat_name"],
                "category": CAT_NAMES[ex["cat"]],
                "label": ex["label"],
                "issue": ex["issue"],
                "source_url": ex["url"],
                "source_doc": ex["source_doc"],
                "page": ex["page"],
                "doc_pages": n_pages,
                "expected": ex["expected"],
                "known_errors": ex["known_errors"],
                "image": str(dest.relative_to(inputs_dir)),
                "width": img.width,
                "height": img.height,
                "orientation": orientation,
                "dpi": a.dpi,
            })

        except Exception as e:
            print(f"       FAILED: {type(e).__name__}: {e}")
            failed.append({
                "eid": ex["eid"],
                "label": ex["label"],
                "url": ex["url"],
                "page": ex["page"],
                "error": f"{type(e).__name__}: {e}",
            })
        print()

    # -- write manifest -----------------------------------------------------
    (inputs_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if failed:
        (inputs_dir / "FAILED.json").write_text(
            json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")

    # -- summary ------------------------------------------------------------
    elapsed = time.time() - t_start
    print("=" * 70)
    print(f"  rendered  {len(manifest)}/{len(examples)}   in {elapsed:.0f}s")

    if manifest:
        by_cat = {}
        for m in manifest:
            by_cat.setdefault(m["cat"], []).append(m)
        print()
        for cid in sorted(by_cat):
            print(f"    cat {cid}  {CAT_NAMES[cid]:<20} {len(by_cat[cid])} pages")
        land = sum(1 for m in manifest if m["orientation"] == "landscape")
        print(f"\n    {land} landscape, {len(manifest) - land} portrait")

    if failed:
        print(f"\n  FAILED {len(failed)}:")
        for f in failed:
            print(f"    {f['eid']:<22} {f['error']}")
            print(f"    {'':<22} {f['url']}")
        print("\n  These would become silent SKIPs during a benchmark run,")
        print("  shrinking the denominator. Fix or drop them first.")

    print(f"\n  manifest -> {inputs_dir / 'manifest.json'}")
    if failed:
        print(f"  failures -> {inputs_dir / 'FAILED.json'}")
    print("=" * 70)

    if a.contact_sheet and manifest:
        print()
        contact_sheet(inputs_dir, [e for e in examples
                                   if any(m["eid"] == e["eid"] for m in manifest)])

    print("\nNext: open a few PNGs and confirm they match the pages in your")
    print("issue-report screenshots. Live documents get republished — if a")
    print("deck gained a slide, page N today is not the page you captured.")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
