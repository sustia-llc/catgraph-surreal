# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/spec/v2.0.0.html). Each version is a git tag; the
crate is not published.

## [0.2.0] - 2026-09-13

### Changed — BREAKING

- catgraph dependencies (`catgraph`, `catgraph-applied`, `catgraph-dl`,
  `catgraph-syntax`) pinned at git tag `v0.23.0` (was `v0.11.0`); the catgraph
  types on this crate's API are the `v0.23.0` types.
- `LineageStore<G>` holds a `TermStore<G>`; `LineageStore::open` bootstraps and
  verifies the term tier's schema along with its own.
- `<usize as LabelCodec>::decode` accepts only the canonical decimal spelling
  (`"007"`, `"+7"`, `" 7"` decode to `None`).

### Added

- `RunRecord::replay(start, rules)` and `LineageStore::replay_run(run)`: a stored
  trace replays through catgraph's `rewrite::replay` against the run's start
  term and its rule set rebuilt in stored order. New runs write
  `replayable = true`; rows written by `0.1.0` keep `false`, and re-recording
  such a run is a no-op that keeps the flag.
- `LabelCodec` for `u8`, `u16`, `u32`, `u64`, `u128`, `i8`, `i16`, `i32`, `i64`,
  `i128`, `isize` (decimal spelling; `decode` accepts the canonical spelling
  only).

### Changed

- `CospanRecord::revalidate` builds through `Cospan::new`, mapping its error to
  `StoreError::Corrupt`; the leg-bounds check on the write and load paths is the
  class derivation (`canonical_classes`).
- Weight rows derive `finite` from `RModule<f64>::is_finite`.
- The test profile builds catgraph with `debug-assertions = false`
  (`[profile.test.package.catgraph]`).

### Fixed

- `decode_coordinates` uses `as_chunks` (clippy 1.98
  `chunks_exact_to_as_chunks`).

## [0.1.0] - 2026-08-13

Initial release: terms, cospans, weights, lineage, documents, and the
notification bus against SurrealDB SDK 3.2.4 and catgraph git tag `v0.11.0`.

[0.2.0]: https://github.com/sustia-llc/catgraph-surreal/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sustia-llc/catgraph-surreal/releases/tag/v0.1.0
