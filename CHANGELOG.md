# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] — 2026-09-18

### Fixed
- `tdr.to_time_domain`: the step response now starts its running sum a few resolution cells before
  t = 0 in signed time, so the half of an edge at the reference plane that the DFT wraps to the end of
  the array is counted. Found while building `zprofile`; the TDR window statistics moved by up to 1 Ω
  on the reference set and no verdict changed. `tests/reference/expected.json` refreshed.
- `gui`: the failure callback captured the exception by reference, which is unbound once the `except`
  block ends; it is now bound as a default argument.

### Added
- `cablecheck demo`: generate the synthetic reference set and evaluate every sample end to end
  (fixture removed, limits applied, reports and database written) in one command.
- `--profile-engine zprofile`: use the loss-aware reconstruction of the companion `zprofile` package
  for the impedance profile and its statistics, when that package is installed.

## [0.1.0] — 2026-09-18

### Added
- First release: Touchstone I/O, network algebra, fixture de-embedding, mixed-mode decomposition,
  quantity computation, limit-line library with standards limit sets, evaluation and verdicts,
  HTML/PDF/JSON reporting, results database, sample pipeline, synthesiser, TDR, CLI and GUI.
- 40 tests and a 13-page technical report.
