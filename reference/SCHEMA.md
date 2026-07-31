# Reference data schema

`build_report.py` renders a report from `outputs/v<model>/results.json` alone.
Supplying a reference file adds a comparison column, verdicts, assessments and
disclaimers.

Reference data is **not committed** — it typically contains prior-version output
transcribed from a third party's bug reports, plus judgement calls about whether
each issue is resolved. Keep it local. `.gitignore` excludes
`reference/*.json`.

## reference.json

Keyed by `eid`, matching the ids in `nemotron_eval.py`. Every field is optional;
cases with no entry render as output-only.

```json
{
  "<eid>": {
    "issue": "The defect as originally described.",

    "verdict": "fixed | part | open | reg",

    "assessment": "Why that verdict, in one or two sentences.",

    "prior_label": "Column heading for the comparison panel.",
    "prior_output": [
      ["Page-header", "text of that element"],
      ["Table", "(first table only)"]
    ],

    "disclaimer": {
      "heading": "Disclaimer — content integrity",
      "kind": "short label shown as a pill",
      "note": "What is wrong with the output beyond the reported issue.",
      "highlight_class": "Chart"
    }
  }
}
```

| Field | Effect |
|---|---|
| `issue` | Italic line under the case title |
| `verdict` | Coloured tag, chip glyph, and summary counts |
| `assessment` | Grey reasoning block under the title |
| `prior_output` | Left-hand Object Classes / Parsed Text panel |
| `prior_label` | Heading for that panel (default "Prior version") |
| `disclaimer` | Red block above the panels |
| `disclaimer.highlight_class` | Rows of that class are highlighted in the output panel |

Verdicts drive the summary tallies. Cases with no `verdict` are counted only in
the page total.

## screenshots.json

Optional. Adds a "Reference screenshot" sub-tab per case.

```json
{
  "<eid>": {
    "src": "where the screenshot came from",
    "b64": "<base64 JPEG, no data: prefix>"
  }
}
```

Embedding images inline keeps the report self-contained, at roughly 60–100 KB
per screenshot. Downscale to about 860 px wide before encoding.

## Usage

```bash
python3.10 build_report.py --model 2.0
python3.10 build_report.py --model 2.0 --reference reference/reference.json
python3.10 build_report.py --model 2.0 \
    --reference reference/reference.json \
    --screenshots reference/screenshots.json \
    --out report/evidence.html
```
