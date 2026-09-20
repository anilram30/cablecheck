# cablecheck — measurement to engineering verdict

**Turns a high-frequency cable measurement into a standards-based engineering verdict and a report.**

[![CI](https://github.com/anilram30/cablecheck/actions/workflows/ci.yml/badge.svg)](https://github.com/anilram30/cablecheck/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-40%20passing-brightgreen)](tests/)
[![Report](https://img.shields.io/badge/report-13%20pages-informational)](docs/report.pdf)

> **Part of the [HF cable toolchain](https://github.com/anilram30/hf-cable-toolchain)** — seven packages that take a high-frequency cable from a raw measurement to a predicted Ethernet link.
> 
> **A · cablecheck**  ·  [B · labauto](https://github.com/anilram30/labauto)  ·  [C · shieldeval](https://github.com/anilram30/shieldeval)  ·  [D · zprofile](https://github.com/anilram30/zprofile)  ·  [E · cableanalytics](https://github.com/anilram30/cableanalytics)  ·  [F · labplatform](https://github.com/anilram30/labplatform)  ·  [G · linktwin](https://github.com/anilram30/linktwin)

---

## The problem it solves

A cable is measured on a vector network analyser and the instrument returns a file of raw scattering parameters. On its own that file answers nothing. `cablecheck` removes the test fixture, converts the raw four-port data into the quantities engineers actually specify (insertion loss, return loss, mode conversion, impedance, delay), compares every one of them against the limit lines of the relevant standard, and issues a pass or fail verdict with the margin and the frequency at which it was worst — then writes the report and files the result.

## At a glance

|  |  |
|---|---|
| **Takes** | Touchstone files from a network analyser (4- or 8-port), a fixture description and a sample record |
| **Produces** | A pass or fail verdict with the margin and the frequency at which it was worst, an HTML/PDF/JSON report, and a row in a results database |
| **Checked against** | Closed-form networks, reciprocity and passivity identities, and a frozen reference set of expected results |
| **Technical report** | [`docs/report.pdf`](docs/report.pdf) — 13 pages, 12 references, every method stated with its mathematics and its limitations |
| **Tests** | 40, run against Python 3.11, 3.12 and 3.13 on every push |
| **Data** | Entirely synthetic. No proprietary or customer measurements are used anywhere in this toolchain. |

## Install

Python 3.11 or newer.

```sh
pip install "git+https://github.com/anilram30/cablecheck.git"
```

---

## What it does

| stage | module | notes |
|---|---|---|
| read raw files | `cablecheck.io` | Touchstone v1 / v2 (any port count, MA/DB/RI, 2-port ordering quirk, noise blocks), VNA CSV dumps |
| remove the fixture | `cablecheck.deembed` | exact T-matrix de-embedding with fixture files; 2x-thru bisection (gated or symmetric, chosen automatically); port extension |
| single-ended → mixed-mode | `cablecheck.mixedmode` | driven by a *port map* such as `A+near,A-near,A+far,A-far`; any number of pairs |
| quantities | `cablecheck.quantities` | insertion loss, return loss, LCL, LCTL, NEXT, FEXT, phase/group delay, delay per metre, skew, NVP, TDR impedance profile, fitted (IEC 61156-1 style) impedance |
| limits | `cablecheck.limits` | cable types as TOML files with closed-form or tabulated limit lines, provenance and status fields |
| verdict | `cablecheck.evaluate` | per-quantity margin (positive = inside limit), worst margin and where it occurs, PASS/FAIL, headline |
| outputs | `cablecheck.report`, `cablecheck.db` | self-contained HTML, PDF, JSON with traces, SQLite (samples / runs / files / results) with SHA-256 of every input |
| interfaces | `cablecheck.cli`, `cablecheck.gui` | `cablecheck evaluate|batch|limits|fixture|db|synth|gui`; Tk interface for lab staff |
| reference data | `cablecheck.synth`, `cablecheck.reference` | multiconductor-transmission-line synthetic cables, fixtures and "measurements" with known physics |

## Install

```bash
pip install -e ".[dev]"
pytest                      # ~40 s, 59 tests incl. golden-file regression
```

## Quick start

```bash
# one sample, fixture removed with an explicit fixture file, HTML + JSON + database
cablecheck evaluate tests/reference/ripple_15m.s4p \
    --cable-type 1000base-t1-link-segment --sample-id ripple_15m --length 15 \
    --fixture-left tests/reference/fixture_launch.s4p \
    --operator "S. Anil" --instrument "VNA-1" --calibration-date 2026-09-17 --temperature 23 \
    --out reports --json --pdf --db results.sqlite

# same with a 2x-thru instead of a fixture file
cablecheck evaluate tests/reference/good_15m.s4p --cable-type 1000base-t1-link-segment \
    --sample-id good_15m --length 15 --thru tests/reference/fixture_2xthru.s4p

# two pairs in one 8-port file (NEXT/FEXT/skew appear automatically)
cablecheck evaluate tests/reference/twopair_5m_truth.s8p --cable-type 1000base-t1-link-segment \
    --sample-id twopair --length 5

# many samples from a job file
cablecheck batch examples/job.json --pdf

# limits, database, fixture tools, GUI
cablecheck limits list
cablecheck limits show 1000base-t1-link-segment --at 1,10,100,600
cablecheck db list --db results.sqlite
cablecheck db export --db results.sqlite --out results.csv
cablecheck fixture tests/reference/fixture_2xthru.s4p -o half.s4p
cablecheck gui
```

Exit code is 0 for PASS and 1 for FAIL, so the command composes with scripts.

## Python API

```python
from cablecheck.pipeline import SampleInfo, MeasurementFile, FixtureSpec, run_sample
from cablecheck.report.html import write_html
from cablecheck.db import ResultsDB

sample = SampleInfo("S-0417", "1000base-t1-link-segment", length_m=15.0, lot="L2026-09",
                    operator="S. Anil", instrument="VNA-1", calibration_date="2026-09-17", temperature_c=23.0)
files = [MeasurementFile("S-0417_pairA.s4p", "A+near,A-near,A+far,A-far",
                         FixtureSpec("files", left="fixture_launch.s4p"))]
result = run_sample(sample, files)
print(result.verdict, result.evaluation.headline.label, result.evaluation.headline.worst_margin)
write_html(result, "S-0417.html")
with ResultsDB("results.sqlite") as db:
    db.insert_run(result, "S-0417.html")
```

## Port maps

One token per VNA port, in port order: `<pair><polarity><end>`, e.g. `A+near,A-near,A+far,A-far`
(the default for 4-port files) or `A+near,A-near,A+far,A-far,B+near,B-near,B+far,B-far`
(default for 8-port files). A 4-port near-end crosstalk file across two pairs is
`A+near,A-near,B+near,B-near`. Unpaired coaxial ports are `se:near` / `se:far`.

## Fixture files

A fixture Touchstone file contains both its VNA-side ports (first half) and its DUT-side
ports (second half). `cablecheck fixture` turns a 2x-thru measurement into such a half.

## Limit files

`src/cablecheck/limits/data/*.toml`. The shipped files are transcribed from *public*
OPEN Alliance / IEEE task-force material and carry `status = "transcribed-unverified"`;
load the lab's controlled limits with `--limit-dir my_limits/`. Each limit has a kind
(`max`, `min`, `range`), frequency segments with an expression in `f` (MHz) or a point
table, optional `per_metre` scaling and `scalar` flags.

## Reference data and regression tests

`tests/reference/` holds five synthetic samples (good, lossy, periodic ripple + defect,
unbalanced, two coupled pairs), the fixture, its 2x-thru, the bare-DUT truth files and
`expected.json`. `pytest tests/test_regression.py` re-evaluates every sample and compares
every reported number; refresh deliberately with `CABLECHECK_UPDATE_EXPECTED=1`.
Regenerate the data with `cablecheck synth --out tests/reference`.

## Documentation

`docs/report.md` / `docs/report.pdf` — full mathematics (power waves, T-parameter
de-embedding, 2x-thru bisection, mixed-mode conversion, the TDR transform and its
choices, fitted impedance, limits and margins), the synthetic model, verification
results and limitations. Build with `docs/build.sh` (pandoc + xelatex).

## Layout

```
pyproject.toml   src/cablecheck/...   tests/   docs/   examples/
```
---

## Contributing

Bug reports, questions about the methods, and pull requests are all welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md). Numerical changes need a numerical test, and a change to a method
is also a change to `docs/report.md`.

## Licence and attribution

MIT — see [LICENSE](LICENSE). Author: Sreeram Anil.

Built with AI assistance; the commit history records it. The engineering decisions, the validation
strategy and the limitations stated in the report are the substance of the work.

Part of the **[HF cable toolchain](https://github.com/anilram30/hf-cable-toolchain)** · [Report an issue](https://github.com/anilram30/cablecheck/issues) ·
[Changelog](CHANGELOG.md)
