# nemotron-parse-20-benchmark

A reproducible harness for evaluating **NVIDIA Nemotron Parse** (2.0, v1.2, v1.1)
on document pages, capturing element classes, parsed text, bounding boxes and
per-page latency.

Given a set of source PDFs and page numbers, it renders each page, runs inference
against a served model, and saves the raw output alongside parsed elements and an
annotated overlay image — so results can be re-scored later without re-running
inference or touching a GPU.

This repository documents the environment and method. It contains no evaluation
results.

---

## Contents

```
prepare_inputs.py     download source PDFs, render the tested page, build a manifest
nemotron_eval.py      run inference, parse output, save elements + bboxes + overlays
score_bugs.py         score saved output offline (no GPU required)
build_report.py       generate a self-contained HTML report from saved output
reference/SCHEMA.md   optional reference-data format for build_report.py
requirements.txt
inputs/               rendered pages + manifest.json  — fixtures
outputs/              per-model results               — generated, gitignored
report/               generated HTML                  — gitignored
```

`inputs/` sits alongside `outputs/`, not inside it. Outputs are regenerated on
every run; inputs are verified fixtures that should survive `rm -rf outputs`.

---

## 1. Hardware and driver

**Start here.** Nemotron Parse 2.0 requires a vLLM build recent enough to
recognise `NemotronParseForConditionalGeneration`. Every such build pins a CUDA 13
PyTorch, which requires **NVIDIA driver 580 or newer**.

| vLLM | torch pinned | Works on driver 570 | Recognises NemotronParse |
|---|---|---|---|
| 0.26.0 | 2.11.0+cu130 | no | yes |
| 0.20.0 | 2.11.0+cu130 | no | yes |
| 0.11.0 | 2.8.0+cu128 | yes | **no** |

No version pair satisfies both constraints, so downgrading vLLM is not a
workaround. Docker is not one either — containers share the host driver; only the
CUDA runtime is containerised. A cu130 image still fails on a 570 host with:

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12080)
```

Check what you have:

```bash
nvidia-smi     # need Driver Version >= 580.x and CUDA Version 13.x
```

### Upgrading the driver

```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-580
```

If the install aborts on a file conflict with leftover `libnvidia-*-570`
packages:

```bash
sudo dpkg -i --force-overwrite /var/cache/apt/archives/libnvidia-gl-580_*.deb
sudo dpkg --configure -a
sudo apt-get install -f -y
sudo apt-get autoremove -y
```

**Verify the kernel module built before rebooting.** Rebooting without a working
module leaves the machine with no GPU and no straightforward remote recovery:

```bash
sudo /usr/sbin/dkms status
# expect: nvidia/580.173.02, <kernel>, x86_64: installed
```

Only then:

```bash
sudo reboot
nvidia-smi     # confirm 580.x / CUDA 13.x after it comes back
```

On some cloud providers "reboot" stops the instance rather than restarting it,
and the root volume may not persist. Confirm the provider's behaviour first.

---

## 2. Python environment

Cloud images often place a virtualenv on `python3` that shadows the interpreter
holding the packages. Identify the right one before installing anything:

```bash
which python3     && python3 --version
which python3.10  && python3.10 --version
which pip         && pip --version
```

Use one interpreter consistently for both `vllm serve` and the scripts. The
examples below assume `python3.10`.

```bash
python3.10 -m pip install --user -r requirements.txt
```

What that pulls and why:

| Package | Reason |
|---|---|
| `vllm==0.26.0` | serving; pins torch 2.11.0+cu130 |
| `timm`, `albumentations`, `open_clip_torch`, `einops` | imported by the model's remote code (C-RADIO vision encoder) |
| `beautifulsoup4` | required by the model repo's `latex2html.py`, which `postprocessing.py` imports |
| `openai`, `pillow`, `requests`, `pypdfium2`, `huggingface_hub` | benchmark harness |

> **Do not use `pip install --force-reinstall` in this environment.** It resolves
> torch upward (2.11 → 2.13), which breaks torchvision and leaves vLLM unable to
> import with `RuntimeError: operator torchvision::nms does not exist`.
> Recovery costs more than whatever it was meant to fix.

---

## 3. Hugging Face access

Public models (`v1.1`, `v1.2`) need only a standard read token. **Parse 2.0 is
private to the `nvidia` org** and needs an explicit organization grant — being an
org member is not sufficient, and the CLI reports a successful login either way.

```bash
hf auth login --token <YOUR_TOKEN>
```

Verify with the raw API rather than `list_models`, which filters to public repos
and will hide the model even when the token is correct:

```bash
curl -s -o /dev/null -w "%{http_code}\n" \
  -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
  https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-Parse-2.0
```

`200` means access is working. `404` means it is not — inspect the token scope:

```bash
curl -s -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
  https://huggingface.co/api/whoami-v2 | python3 -m json.tool | grep -A6 '"scoped"'
```

An entry for the nvidia entity with `"permissions": []` means the org is attached
to the token but no permissions were granted. On huggingface.co → Settings →
Tokens, edit the token and scroll **past** the *User permissions* block to
*Organization permissions → nvidia*, then enable read access to repository
contents. A classic **Read** token avoids this entirely, since it inherits org
access without per-scope configuration.

---

## 4. Model repo helpers

Output parsing must use the model's own `extract_classes_bboxes`. Classes and
coordinates are emitted as special tokens (`<x_0.1592><y_0.1039>`), not plain
text, so a hand-written regex will silently extract nothing.

```bash
python3.10 nemotron_eval.py --fetch-postprocessing
```

This fetches every top-level `.py` from the model repo, not just
`postprocessing.py`, because that file imports `latex2html.py` as a sibling.

Confirm:

```bash
python3.10 -c "from postprocessing import extract_classes_bboxes; print('ok')"
```

---

## 5. Tied-embedding patch — required for 2.0 only

The 2.0 export keeps `lm_head.weight` tied to `decoder.embed_tokens.weight`.
Current vLLM builds materialise a separate, randomly-initialised output head
unless patched.

**Without the patch there is no error.** The model loads, weights load, the
server starts, and generation returns fluent nonsense — `<s><s><s>…` or
`ountountount…` repeated to the token limit. If you see that, this is the cause.

```bash
PATCH_ROOT=$(python3.10 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('nvidia/NVIDIA-Nemotron-Parse-2.0',
      allow_patterns='vllm_tied_patch/sitecustomize.py'))")
export PYTHONPATH="${PATCH_ROOT}/vllm_tied_patch:${PYTHONPATH:-}"

echo "$PYTHONPATH"
ls -l "${PATCH_ROOT}/vllm_tied_patch/"
```

`PYTHONPATH` only affects processes started afterwards — export it in the same
shell **before** `vllm serve`. It cannot be applied to a running server. Not
required for v1.1 or v1.2.

---

## 6. Prepare inputs

```bash
python3.10 prepare_inputs.py --contact-sheet
```

Downloads each source PDF once (cached in `/tmp/nemotron_pdfs`), renders the page
under test at 150 DPI, writes `inputs/manifest.json`, and reports any URL that
failed.

Run this **before** benchmarking. During a run a dead URL becomes a silent skip
that shrinks the denominator without any obvious signal.

Useful flags:

```bash
--dpi 200          # higher render resolution
--cat 4            # one category only
--force            # ignore cache, re-download
--check            # verify already-rendered inputs, no network
--contact-sheet    # labelled montage of every input for review
```

Open `inputs/contact_sheet.png` and confirm each render is the page you intended.
Source documents are live and get republished; page N today may not be page N
when the reference was captured.

---

## 7. Serve

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1

CHAT_TEMPLATE=$(python3.10 -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('nvidia/NVIDIA-Nemotron-Parse-2.0','chat_template.jinja'))")

vllm serve nvidia/NVIDIA-Nemotron-Parse-2.0 \
    --dtype bfloat16 \
    --max-num-seqs 1 \
    --limit-mm-per-prompt '{"image": 1}' \
    --trust-remote-code \
    --port 8000 \
    --attention-backend TRITON_ATTN \
    --chat-template "${CHAT_TEMPLATE}"
```

Notes on each choice:

- **`--chat-template`** takes a path. The file ships inside the model repo, not
  the working directory, so it must be resolved first.
- **`--max-num-seqs 1`** keeps inference to one request at a time, so latency
  reflects single-page cold-start rather than batched throughput.
- **`--attention-backend TRITON_ATTN`** is NVIDIA's recommendation for A100 and
  A10 systems.
- **`FLASHINFER_DISABLE_VERSION_CHECK=1`** works around a mismatch between
  `flashinfer-python` and `flashinfer-cubin` versions. Safe here: the guard
  protects top-k/top-p sampler kernels and this harness decodes greedily
  (`temperature 0`, `top_k 1`). Do not attempt a pip fix — the CUDA dependency
  tree backtracks for a long time and still fails to resolve.

vLLM will log `torch.compile is turned on, but the model does not support it` and
drop cudagraphs to `FULL_DECODE_ONLY`. That is expected for this architecture and
applies equally to all three versions.

Wait for `Application startup complete`, then in another terminal:

```bash
curl -sf http://localhost:8000/health && echo healthy
```

### Serving v1.2 and v1.1

Same command with the model id changed. Neither needs the tied-embedding patch.
v1.1 has no chat template in-repo; omit the flag unless vLLM asks for one.

---

## 8. Self-test before a full run

```bash
python3.10 nemotron_eval.py --selftest --model 2.0
```

Runs a single page and prints the raw model output next to what the parser
extracted, including bounding boxes in pixel coordinates.

If it reports zero parsed elements, **stop**. A full run would score every page
0% and look like a model failure rather than a parsing failure. Capture the raw
block and fix the parser first.

---

## 9. Run

```bash
python3.10 nemotron_eval.py --model 2.0
python3.10 nemotron_eval.py --model 1.2
python3.10 nemotron_eval.py --model 1.1
```

Stop and re-serve between models. Options:

```bash
--limit 3                                # smoke test
--base-url http://localhost:8000/v1      # non-default endpoint
--out-dir ./outputs                      # default
```

Then score offline:

```bash
python3.10 score_bugs.py --detail
python3.10 score_bugs.py --csv results.csv
```

## 10. Build a report

```bash
python3.10 build_report.py --model 2.0
```

Writes `report/evidence-v2.0.html` — self-contained, no CDN, printable with all
tabs expanded. A summary tab plus one tab per category, each with per-case chips
and an Object Classes / Parsed Text panel.

With output alone the report shows what the model returned. To add a comparison
column, verdicts, assessments or disclaimers, supply a reference file:

```bash
python3.10 build_report.py --model 2.0 --reference reference/reference.json
```

See `reference/SCHEMA.md` for the format. Reference data is gitignored — it
usually holds prior-version output and judgement calls, which are separate from
the harness itself.

---

## Prompt formats

| Purpose | Prompt |
|---|---|
| Default (2.0, v1.2) | `</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>` |
| Extract text inside figures | `</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>` |
| Classes and boxes only | `</s><s><predict_bbox><predict_classes><output_no_text><predict_no_text_in_pic>` |
| v1.1 (three-token, legacy) | `</s><s><predict_bbox><predict_classes><output_markdown>` |

v1.2 introduced the fourth token. The prompt is not validated at runtime, so
using the v1.1 form against v1.2 or 2.0 produces output — but all v1.2+ training
data used four tokens, and quality degrades noticeably without it.
`nemotron_eval.py` selects the correct prompt per model.

`skip_special_tokens=False` is required in the request, because classes and
coordinates *are* special tokens.

Sampling used here: `temperature 0`, `top_k 1`, `repetition_penalty 1.1`,
`max_tokens 8192`. Note the context limit is 9000 total — `max_tokens` must leave
room for the prompt, or the request is rejected.

---

## Output layout

```
outputs/v2.0/
  cat1_misclassification/
    <case>.json          elements: class, parsed text, bbox (normalised + pixel)
    <case>_raw.txt       unmodified model output
    <case>_overlay.png   boxes drawn on the page, coloured and numbered by class
  …
  results.json           all records for this model
  run_summary.txt        element counts and latency
```

Per-element record:

```json
{
  "cls": "Table",
  "text": "\\begin{tabular}{ccccc} ...",
  "bbox":    [0.10, 0.12, 0.90, 0.45],
  "bbox_px": [127, 198, 1147, 742]
}
```

Latency is captured with `time.perf_counter()` around the API call only, so PDF
download and rendering do not contaminate it:

```json
{ "total_s": 1.07, "tokens_out": 582, "tokens_per_sec": 544.9 }
```

Raw output is retained per page so scoring can be revised without re-running
inference.

---

## Scoring approach

Positional class-matching against hand-written expected lists was tried and
abandoned. Two reasons:

1. Real pages carry 6–40 elements. Reference screenshots typically show only the
   first few, so any hand-written expectation is far too short.
2. Positional alignment is brittle — one extra element at index 0 shifts
   everything after it, turning a correct parse into a 0% score.

`score_bugs.py` instead evaluates targeted signatures: whether captions are
orphaned from any figure, whether `Page-Footer` is last, whether an expected
number of tables was found, whether body content survived. Each check returns
PASS / FAIL / N/A and runs entirely against saved JSON.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Output is `<s><s><s>…` or `ountount…` | Tied-embedding patch not applied. Export `PYTHONPATH` **before** `vllm serve`. |
| `RepositoryNotFound` for 2.0 | Token lacks the nvidia **organization** grant. User-scope permissions are not enough. |
| `driver is too old (found version 12080)` | Driver older than 580 against a cu130 torch. Upgrade the driver; downgrading vLLM does not work. |
| `flashinfer-cubin … does not match flashinfer` | `export FLASHINFER_DISABLE_VERSION_CHECK=1`. Do not attempt a pip fix. |
| `No module named 'timm'` | Model remote-code dependency. See `requirements.txt`. |
| `No module named 'bs4'` | Needed by the repo's `latex2html.py`. Install `beautifulsoup4`. |
| `chat_template.jinja … doesn't exist` | Pass the resolved repo path, not a bare filename. |
| `maximum context length is 9000 tokens` | `max_tokens` must leave room for the prompt. This harness uses 8192. |
| `postprocessing.py not found` | Its own import failed. Run `--fetch-postprocessing` to fetch all sibling `.py` files. |
| `operator torchvision::nms does not exist` | A `--force-reinstall` pulled torch to 2.13. Pin back: `torch==2.11.0 torchvision==0.26.0`. |
| Parser returns 0 elements | Do not proceed. Inspect the `--selftest` raw output and match the parser to the actual format. |
| 403 when fetching a source PDF mid-run | Some hosts rate-limit repeat requests. `nemotron_eval.py` reads `inputs/` first — run `prepare_inputs.py` beforehand. |
| `zip: command not found` | Use `tar -czf`, or upload files individually. |

---

## Reference environment

The configuration this harness was developed against:

| | |
|---|---|
| GPU | NVIDIA H100 PCIe 80GB |
| Driver | 580.173.02 |
| CUDA | 13.0 |
| Python | 3.10 |
| vLLM | 0.26.0 |
| torch | 2.11.0+cu130 |
| transformers | 5.14.1 |
| Render | pypdfium2, 150 DPI |

---

## Licence notes

The model's own `.py` helpers (`postprocessing.py`, `latex2html.py`, `hf_*.py`)
are fetched at setup rather than vendored here — they are NVIDIA's, under the
NVIDIA Open Model License. `.gitignore` excludes them.

Source documents used as inputs belong to their respective publishers. Check
their terms before committing rendered pages to a public repository.
