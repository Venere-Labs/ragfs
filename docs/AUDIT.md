# RAGFS Audit & Improvement Tracker

This file is the single home for audit findings, so that they do not exist only
inside pull-request descriptions. **Update it in the same PR that changes the
underlying fact.** If a row here disagrees with the code, the code wins and the
row is a bug.

Last updated: 2026-09-15.

---

## 1. Dependency security posture

Measured with `cargo audit` against a local RustSec advisory-db snapshot. The
scheduled `Security Audit` workflow uses a freshly fetched database and can
report a slightly different set of IDs than the numbers below.

| Metric | Before | Now |
|---|---:|---:|
| Vulnerabilities | 28 | **0** |
| Unsound (entries) | 8 | **0** |
| Unmaintained (entries) | 8 | 6 (all upstream-pinned) |

**There is no `ignore:` list anywhere.** The workflow that used to carry one now
runs with no suppressions at all.

Before this work the workflow had also been failing on `main` on **every
scheduled run since at least 2026-05** (30+ consecutive failures) for a second,
independent reason: `rustsec/audit-check` aborted with `Resource not accessible
by integration` because the job lacked `issues: write`, so it produced no
actionable report at all. See §4.

`main` additionally did not compile at all on rustc 1.97 because of
`ethnum 1.5.2`; that is fixed too, and it is what kept every CI job red.

### 1.1 Resolved

| Advisory | Crate | From → To | How |
|---|---|---|---|
| RUSTSEC-2026-0005 | oneshot | 0.1.11 → 0.1.13 | lock |
| RUSTSEC-2026-0007 | bytes | 1.11.0 → 1.12.1 | lock |
| RUSTSEC-2026-0009 | time | 0.3.44 → 0.3.47 | lock |
| RUSTSEC-2026-0037, -0185 | quinn-proto | 0.11.13 → 0.11.18 | lock |
| RUSTSEC-2026-0041 | lz4_flex | 0.11.5 → 0.11.6 | lock |
| RUSTSEC-2026-0044 … -0048 | aws-lc-sys | 0.35.0 → 0.45.0 | lock |
| RUSTSEC-2026-0049, -0098, -0099, -0104 | rustls-webpki | 0.103.8 → 0.103.15 | lock |
| RUSTSEC-2026-0097 | rand | 0.8.5 → 0.8.8, 0.9.2 → 0.9.5 | lock |
| RUSTSEC-2026-0186 | memmap2 | 0.9.9 → 0.9.11 | lock |
| RUSTSEC-2026-0190 | anyhow | 1.0.100 → 1.0.104 | lock |
| RUSTSEC-2026-0204 | crossbeam-epoch | 0.9.18 → 0.9.21 | lock |
| RUSTSEC-2026-0221 | event-listener | 5.4.1 → 5.4.2 | lock |
| RUSTSEC-2026-0258 | h2 (0.4 line) | 0.4.13 → 0.4.19 | lock |
| RUSTSEC-2026-0187 | lopdf | 0.38.0 → 0.42.0 | code + manifest |
| RUSTSEC-2026-0176, -0177 | pyo3 | 0.27.2 → 0.29.2 | code + manifest |
| RUSTSEC-2025-0069 | daemonize | 0.5.0 → removed | code + manifest |
| RUSTSEC-2026-0258, -0194, -0195, -2023-0071, -0098, -0099, -0104 | h2 0.3, quick-xml, rsa, rustls-webpki 0.101 | removed from the graph | `lancedb` 0.23 → 0.37.1 |
| (build blocker) | ethnum | 1.5.2 → 1.5.3 | lock |

Detail on the changes that needed code:

- **lancedb.** This is what actually cleared the last nine advisories. The
  `lancedb` 0.23 generation pulled `lance → aws-sdk-dynamodb →
  aws-smithy-http-client` (h2 0.3, rustls-webpki 0.101), `lance-io → opendal →
  reqsign` (quick-xml 0.37, rsa) and `object_store` 0.12 (quick-xml 0.38,
  rustls-pemfile). Moving to 0.37.1 (arrow 58, lance 10, datafusion 54,
  `object_store` 0.13) drops all of those chains. The only code change needed was
  coercing the batch reader for `Table::add`, which now requires `Scannable`:
  `Box::new(batches) as Box<dyn RecordBatchReader + Send>`.
  **Do not jump to 0.38.0 yet**: it does not compile with its default (empty)
  feature set, because `src/job.rs` uses `Error::Http` while that variant is
  `#[cfg(feature = "remote")]`. 0.37.1 has no such reference.
- **lopdf.** `pdf-extract` 0.10 pulled `lopdf ^0.38`, so bumping only our direct
  dependency would have left a second, vulnerable copy in the graph. Both moved:
  `pdf-extract` 0.10 → 0.12 and `lopdf` 0.38 → **0.42**, deliberately not 0.45,
  so that exactly one lopdf resolves. `PdfImage` gained a lifetime parameter in
  this range, and the extraction path was rewritten around it.
- **pyo3.** 0.28 deprecates the automatic `FromPyObject` for `#[pyclass]` types
  that derive `Clone`. The 5 types used as arguments (`PyOrganizeRequest`,
  `PyOrganizeStrategy`, `PyOperation`, `Document`, `PyChunk`) now opt in with
  `#[pyclass(from_py_object)]`; the 22 output-only types declare
  `#[pyclass(skip_from_py_object)]`, which becomes the default in a later release.
- **daemonize.** Replaced with `nix` `fork`/`setsid`/`dup2`. Daemonization now
  happens **before** the Tokio runtime is built, because forking a process that
  already owns worker threads is not safe; the old code forked from inside
  `#[tokio::main]`. The log file is opened once instead of twice (the second
  `File::create` used to truncate the first).

**Also fixed here, not advisory-driven:** the PDF `FlateDecode` path called
`read_to_end` on untrusted zlib data and only checked the 50 MB cap *after*
decoding, i.e. a decompression bomb. Images are now read against the remaining
byte budget and rejected when larger. Covered by a regression test.

### 1.2 Unmaintained — no fix available from this repository

These do not fail the audit (`unmaintained` is informational), and none of them
can be removed without dropping a dependency we deliberately use. Each row names
the reason and what would have to change upstream.

| Advisory | Crate | Pulled by | Why it stays |
|---|---|---|---|
| RUSTSEC-2024-0436 | paste 1.0.15 | `ragfs-embed → tokenizers` | `tokenizers` 0.23.2 still requires `paste ^1.0.14`; no newer tokenizers exists |
| RUSTSEC-2025-0119 | number_prefix 0.4.0 | `ragfs-embed → hf-hub → indicatif` | `hf-hub` 0.4.3 pulls `indicatif` unconditionally through **both** its `tokio` and `ureq` features, so it cannot be switched off; `hf-hub` 1.0 drops it but is a breaking rewrite of the model-download path |
| RUSTSEC-2026-0105 | core2 0.4.0 | `ragfs-extract → pdf_oxide → libflate` | Optional `pdf_oxide` extractor, deliberately added by the maintainers |
| RUSTSEC-2020-0144 | lzw 0.10.0 | `ragfs-extract → pdf_oxide` | Same |
| RUSTSEC-2026-0192 | ttf-parser 0.24.1 | `ragfs-extract → pdf_oxide` | Same |
| RUSTSEC-2026-0192 | ttf-parser 0.25.1 | `ragfs-extract → lopdf 0.42` | `lopdf` 0.45 replaced `ttf-parser` with `skrifa`, but moving to it re-introduces a second lopdf unless `pdf-extract` is also dropped |

If the `pdf_oxide` extractor is ever retired (see issue #58), three of these rows
disappear with it.

### 1.3 Review

Nothing here needs a review date any more, because nothing is suppressed. Re-run
§5 periodically; the corpus is clean, so any new ID is a genuine regression.

---

## 2. Product audit must-fixes (1–10)

Delivered by the open PR stack. The audit itself previously existed only inside
PR descriptions; this table is its durable home.

| # | Finding | PR |
|---|---|---|
| 1–5, 9 | `config.toml` was inert; `--force` ignored; `index` did not wait for idle; `mount` did not scan/watch; hybrid not wired; default model was `jina-embeddings-v3` instead of the implemented `thenlper/gte-small` | #41 |
| 6 | `SearchQuery.filters` (lang/path/type/depth) and `DistanceMetric` ignored by `LanceStore` | #38 |
| 7–8 | No path jail for agent operations; `.ops/.result` `undo_id` was not the history id | #40 |
| 10 | MCP computed `indices/default` instead of the CLI's `indices/blake3[:16]` scheme | #39 |

CI unblocking: #42 (narrow, base of the stack) vs #37 (broad, also rewrites the
Python integrations). They conflict and **only one should land** — #42 is
recommended. Merge order: **#42 → #40 → #41 → #38 → #39**.

---

## 3. Open work items

### P0

- [x] Fix the three direct-dependency advisories (lopdf, pyo3, daemonize) — issue #44.
- [x] Make `main` compile on current rustc (`ethnum 1.5.3`).
- [x] Migrate `ragfs-store` to `lancedb` 0.37 to clear the last nine
      advisories; the `ignore:` list is now empty everywhere.
- [ ] Fix the `deny.toml` ignore list in #42: it still suppresses 22 IDs,
      including ones that no longer appear in the graph at all, mislabels
      `RUSTSEC-2026-0098`/`-0099` as `pyo3` (they are `rustls-webpki`), and relaxes
      `wildcards` from `deny` to `allow`. With this branch it can be emptied.
      Issue #45.
- [ ] Unblock and land the merge order in §2, then re-run release PR #13. Issue #46.
- [ ] Triage the 18 Dependabot PRs as one grouped change. Issue #47.
- [ ] Report the `lancedb` 0.38.0 build failure upstream (`job.rs` uses
      `Error::Http` without the `remote` feature) so the pin can be lifted.

### P1

- [x] Create GitHub issues for the must-fixes — #44–#53; the repository had zero.
- [ ] Align stale documentation with the code: `README.md` claims "270+ tests"
      and `CHANGELOG.md` "291", while the tree contains 390 test functions;
      `CLAUDE.md` says "9 crates" while there are 10 Rust crates plus the Python
      `ragfs-mcp` crate and the framework adapters. Issue #48.
- [ ] Single source of truth for the embedding model; fail fast on a store/model
      dimension mismatch. Issue #49.
- [ ] Propagate the path jail beyond `ragfs-fuse` (`ragfs-python`, `ragfs-mcp`).
      Issue #50.
- [ ] CLI integration tests for config precedence, `--force`, `--hybrid`. Issue #51.
- [ ] Split `crates/ragfs-fuse/src/filesystem.rs` (2 189 lines). Issue #52.
- [ ] Decide the fate of the `pdf_oxide` feature (§1.2). Issue #58.

### P2

- [ ] Nightly `cargo bench` smoke run — `benches/` exists but never executes.
- [ ] Add a `patch` coverage gate in `codecov.yml` for PRs.
- [ ] Document an advisory-response policy in `SECURITY.md` with a review cadence.

---

## 4. CI notes

- **`security.yml` permissions.** `issues: write` is required because
  `rustsec/audit-check` files its report as an issue. Its absence made the
  workflow fail *after* a successful scan, on every run, producing no report.
  The same fix is applied to the `security-audit` job in `pre-release.yml`.
- **`security.yml` no longer runs on `pull_request`.** PR-time advisory gating is
  already done by the `cargo-deny` job in `ci.yml`. Running the full audit on PRs
  as well would turn every PR red for lockfile advisories it did not introduce.
- **No `ignore:` lists remain.** They were only ever a temporary measure while
  §1.2-style constraints existed; the graph is now clean, so nothing is
  suppressed in `security.yml`, `pre-release.yml` or (once #42 is revised)
  `deny.toml`.
- **`RUSTFLAGS: -Dwarnings` is no longer global.** It used to sit in the
  workflow-level `env`, which made `check`, `test`, `msrv` and `doc` fail in
  lockstep on every new rustc warning. Warnings remain a hard error in the
  `clippy` job (and `doc` for rustdoc).
- **Concurrency.** `ci.yml` had no `concurrency` block, so every push queued
  another ~15-job run. With 24 open PRs this starved the runners: open PRs sat
  in `queued` for 6–24 minutes with no check ever completing. Added
  `cancel-in-progress` to `ci.yml` and `security.yml`.
- **Draft PRs.** `doc`, `msrv`, `python-integration` and the 3.10–3.13 Python
  matrix now run only on `main` and on ready-for-review PRs. Drafts still get
  `check`, `fmt`, `clippy`, `test`, `cargo-deny`, `python-lint` and one Python
  interpreter.

---

## 5. How to reproduce

```bash
# cargo-audit needs a writable CARGO_HOME; ~/.cargo may be read-only.
export CARGO_HOME=/tmp/ragfs-cargo-home

cargo audit --no-fetch --db ~/.cargo/advisory-db            # human-readable
cargo audit --no-fetch --db ~/.cargo/advisory-db --json     # machine-readable

# Reverse dependency paths for any advisory, straight from Cargo.lock
cargo tree -i <crate>@<version>

# Full policy check (advisories + licenses + bans), as CI runs it
cargo deny --all-features check
```

Build note: this machine has `rustc-wrapper = "/usr/bin/sccache"` in
`~/.cargo/config.toml`. sccache resolves `rustc` through the rustup shim, which
tries to install a toolchain into the read-only `~/.rustup`. For local builds,
unset the wrapper and pin the compiler:

```bash
export RUSTC_WRAPPER=""
export RUSTC=~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/rustc
```

Test-function count quoted in §3:

```bash
grep -rn '#\[tokio::test\]\|#\[test\]' --include=*.rs crates | wc -l   # 390
```
