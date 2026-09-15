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

| Metric | Before | After lockfile remediation |
|---|---:|---:|
| Vulnerabilities | 28 | **12** |
| Unsound (entries) | 8 | **2** |
| Unmaintained (entries) | 8 | 8 |

The workflow itself had been failing on `main` on **every scheduled run since at
least 2026-05** (30+ consecutive failures) for two independent reasons: 26 open
advisories, and `rustsec/audit-check` aborting with `Resource not accessible by
integration` because the job lacked `issues: write`. It therefore produced no
actionable report at all. See §4.

### 1.1 Resolved by lockfile updates only

All of these were fixed with `cargo update -p <crate>` inside the ranges the
current dependency generation already allowed. No source or manifest change was
required.

| Advisory | Crate | From → To | Kind |
|---|---|---|---|
| RUSTSEC-2026-0005 | oneshot | 0.1.11 → 0.1.13 | unsound (use-after-free) |
| RUSTSEC-2026-0007 | bytes | 1.11.0 → 1.12.1 | vulnerability |
| RUSTSEC-2026-0009 | time | 0.3.44 → 0.3.55 | vulnerability |
| RUSTSEC-2026-0037, -0185 | quinn-proto | 0.11.13 → 0.11.18 | vulnerability (DoS) |
| RUSTSEC-2026-0041 | lz4_flex | 0.11.5 → 0.11.6 | vulnerability |
| RUSTSEC-2026-0044 … -0048 | aws-lc-sys | 0.35.0 → 0.45.0 | vulnerability |
| RUSTSEC-2026-0049, -0098, -0099, -0104 | rustls-webpki | 0.103.8 → 0.103.15 | vulnerability |
| RUSTSEC-2026-0186 | memmap2 | 0.9.9 → 0.9.11 | unsound |
| RUSTSEC-2026-0190 | anyhow | 1.0.100 → 1.0.104 | unsound (`downcast_mut` UB) |
| RUSTSEC-2026-0204 | crossbeam-epoch | 0.9.18 → 0.9.21 | vulnerability |
| RUSTSEC-2026-0221 | event-listener | 5.4.1 → 5.4.2 | unsound |
| RUSTSEC-2026-0097 | rand | 0.8.5 → 0.8.8, 0.9.2 → 0.9.5 | unsound |
| RUSTSEC-2026-0258 | h2 (0.4 line) | 0.4.13 → 0.4.19 | vulnerability |

`aws-lc-rs` was bumped 1.15.2 → 1.18.1 alongside `aws-lc-sys`.

**Supply-chain note.** These updates pulled a handful of new transitive packages
into the graph (`chacha20`, `cpufeatures`, `getrandom 0.4`, `r-efi`,
`rand 0.10`/`rand_core 0.10`/`rand_pcg 0.10`, plus `deranged`/`num-conv`/
`time-core`/`time-macros` bumps). They are the upstream-proposed fix for
`quinn-proto` and `rand`; they are not a hand-written selection. A blanket
`cargo update` was deliberately **not** run — it would have moved unrelated
crates across major versions (`zip 1.1.4 → 7.2.0` among others).

### 1.2 Residual vulnerabilities — accepted, with justification

These are listed in the `ignore:` input of both `.github/workflows/security.yml`
and `.github/workflows/pre-release.yml`. The admission rule is strict: an ID
belongs here only when **no upgrade exists inside the ranges the current
dependency generation allows**, i.e. the fix requires an upstream or a major
dependency move that is a project of its own.

| Advisory | Crate | Current | Patched | Dependency path | Reason accepted |
|---|---|---|---|---|---|
| RUSTSEC-2026-0258 | h2 | 0.3.27 | ≥0.4.16 | `ragfs-store → lancedb → lance → aws-sdk-dynamodb → aws-smithy-http-client` | Legacy 0.3 line; only reachable through the AWS/DynamoDB backend, which RAGFS does not use by default |
| RUSTSEC-2026-0194, -0195 | quick-xml | 0.37.5 | ≥0.41.0 | `ragfs-store → lancedb → lance-io → opendal → reqsign` | Needs an `opendal`/`reqsign` upgrade driven by a `lancedb` upgrade |
| RUSTSEC-2026-0194, -0195 | quick-xml | 0.38.4 | ≥0.41.0 | `ragfs-store → lancedb → object_store` | Needs an `object_store` upgrade driven by a `lancedb` upgrade |
| RUSTSEC-2023-0071 | rsa | 0.9.10 | none | `ragfs-store → lancedb → lance-io → opendal → reqsign` | No upstream patch exists; RAGFS performs no RSA private-key operation |
| RUSTSEC-2026-0098, -0099, -0104 | rustls-webpki | 0.101.7 | ≥0.103.12 | `ragfs-store → lancedb → lance → aws-sdk-dynamodb → rustls 0.21` | Legacy rustls 0.21 line, pinned by the AWS SDK chain |

Common remedy for the whole table: move to a `lancedb`/`arrow` generation where
these transitive pins are gone. That upgrade is already proposed as Dependabot
#26/#28 (`arrow-schema`/`arrow-array` 57) and must be treated as security work,
not as a routine bump.

**Review by: 2026-12-31.** When reviewing, re-run §5 and delete any row that no
longer appears.

### 1.3 Residual vulnerabilities — NOT accepted, P0 work

These are **intentionally absent** from the `ignore:` list, so the scheduled
audit stays red until they are fixed. That is the intended pressure.

| Advisory | Crate | Current | Patched | Path | Why it matters | Action |
|---|---|---|---|---|---|---|
| RUSTSEC-2026-0187 | lopdf | 0.38.0 | **≥0.42.0** | *direct*: `ragfs-extract → lopdf` | RAGFS parses untrusted PDFs as a core feature; DoS (`AV:N/AC:L`, availability High) | Bump `lopdf` to `0.42` in the workspace `Cargo.toml`, adapt `crates/ragfs-extract/src/pdf.rs`, re-run extractor tests |
| RUSTSEC-2026-0176, -0177 | pyo3 | 0.27.2 | **≥0.29.0** | *direct*: `ragfs-python → pyo3` | Python bindings surface | Bump `pyo3`/`pyo3-async-runtimes` to `0.29`, migrate `crates/ragfs-python` |
| RUSTSEC-2025-0069 | daemonize | 0.5.0 | unmaintained | *direct*: `ragfs → daemonize` | Used by `ragfs mount` daemonization; upstream archived, has an open UB fix | Replace with `rustix`/`nix` `fork` + `setsid`, or vendor a maintained fork |

### 1.4 Unmaintained / unsound (warnings, do not fail the audit)

| Advisory | Crate | Path | Note |
|---|---|---|---|
| RUSTSEC-2026-0002, -0253 | lru 0.12.5 | `lancedb → lance → tantivy` | Unsound; needs `lru ≥0.16.3` via upstream |
| RUSTSEC-2024-0436 | paste 1.0.15 | `ragfs-embed → tokenizers` | Unmaintained |
| RUSTSEC-2026-0105 | core2 0.4.0 | `ragfs-extract → pdf_oxide → libflate` | Unmaintained (also yanked) |
| RUSTSEC-2020-0144 | lzw 0.10.0 | `ragfs-extract → pdf_oxide` | Unmaintained |
| RUSTSEC-2025-0119 | number_prefix 0.4.0 | `ragfs-embed → hf-hub → indicatif` | Unmaintained |
| RUSTSEC-2025-0134 | rustls-pemfile 2.2.0 | `ragfs-store → lancedb → object_store` | Unmaintained |
| RUSTSEC-2026-0192 | ttf-parser 0.24.1, 0.25.1 | `ragfs-extract → lopdf` | Unmaintained |

---

## 2. Product audit must-fixes (1–10)

Delivered by the open PR stack. The audit itself previously existed only inside
PR descriptions; this table is its durable home.

| # | Finding | PR |
|---|---|---|
| 1–5, 9 | `config.toml` was inert; `--force` ignored; `index` did not wait for idle; `mount` did not scan/watch; hybrid not wired; default model was `jina-embeddings-v3` instead of the implemented `thenlper/gte-small` | #41 |
| 6 | `SearchQuery.filters` (lang/path/type/depth) and `DistanceMetric` ignored by `LanceStore` | #38 |
| 7–8 | No path jail for agent operations; `.ops/.result` `undo_id` was not the history id | #40 |
| 10 | MCP computed `indices/default` instead of the CLI's `indices/{blake3[:16]}` scheme | #39 |

CI unblocking: #42 (narrow, base of the stack) vs #37 (broad, also rewrites the
Python integrations). They conflict and **only one should land** — #42 is
recommended. Merge order: **#42 → #40 → #41 → #38 → #39**.

---

## 3. Open work items

### P0

- [ ] Trim the `ignore:` list in #42's `deny.toml` to the rows in §1.2. It
      currently suppresses 22 IDs including fixable ones (`RUSTSEC-2026-0194`,
      `-0195`, `RUSTSEC-2023-0071`, `RUSTSEC-2025-0069`) with one mislabelled
      comment (`RUSTSEC-2026-0098`/`-0099` are `rustls-webpki`, not `pyo3`), and
      it relaxes `wildcards` from `deny` to `allow`.
- [ ] Fix the three direct-dependency advisories in §1.3.
- [ ] Unblock and land the merge order in §2, then re-run release PR #13.
- [ ] Triage the 18 Dependabot PRs as one grouped change; treat
      `arrow`/`lancedb` as security work (§1.2).

### P1

- [ ] Create GitHub issues for the must-fixes and the items in this file — the
      repository currently has **zero** issues, so nothing here is trackable
      outside PRs.
- [ ] Align stale documentation with the code: `README.md` claims "270+ tests"
      and `CHANGELOG.md` "291", while the tree contains 390 test functions;
      `CLAUDE.md` says "9 crates" while there are 10 Rust crates plus the Python
      `ragfs-mcp` crate and the Python framework adapters.
- [ ] Single source of truth for the embedding model. `MODEL_ID`/`EMBEDDING_DIM`
      are duplicated in `crates/ragfs-embed/src/candle.rs` and
      `crates/ragfs/src/main.rs`, and `LanceStore::new(db_path, EMBEDDING_DIM)`
      silently assumes they agree. Derive both from `Embedder`, and fail fast on
      a store/model dimension mismatch.
- [ ] Propagate the path jail beyond `ragfs-fuse`. The same operations are
      reimplemented in `crates/ragfs-python/src/{ops,semantic}.rs` and
      `crates/ragfs-mcp/src/ragfs_mcp/server.py`, so #40's fix does not protect
      those entry points. Extract the shared logic and have MCP delegate to the
      PyO3 layer instead of reimplementing it.
- [ ] Add CLI integration tests for the behaviour #41 introduces (config
      precedence, `--force`, `--hybrid`) using `NoopEmbedder`/`MemoryStore`, so
      they run without a model download.
- [ ] Split `crates/ragfs-fuse/src/filesystem.rs` (2 189 lines) by
      responsibility. Do it after the stack lands to avoid conflicts.

### P2

- [ ] Nightly `cargo bench` smoke run — `benches/` exists but never executes.
- [ ] Add a `patch` coverage gate in `codecov.yml` for PRs.
- [ ] Document an advisory-response policy in `SECURITY.md` (vulnerability =
      blocker, unmaintained = issue) with a review cadence.

---

## 4. CI notes

- **`security.yml` permissions.** `issues: write` is required because
  `rustsec/audit-check` files its report as an issue. Its absence made the
  workflow fail *after* a successful scan, on every run, producing no report.
  The same fix is applied to the `security-audit` job in `pre-release.yml`.
- **`security.yml` no longer runs on `pull_request`.** PR-time advisory gating is
  already done by the `cargo-deny` job in `ci.yml`. Running the full audit on PRs
  as well would turn every PR red for lockfile advisories it did not introduce.
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
- **Duplication to keep in sync.** The `ignore:` list exists in three places:
  `security.yml`, `pre-release.yml` and `deny.toml`. Prefer `deny.toml` as the
  source of truth when consolidating.

---

## 5. How to reproduce

```bash
# cargo-audit needs a writable CARGO_HOME; ~/.cargo may be read-only.
export CARGO_HOME=/tmp/ragfs-cargo-home

cargo audit --no-fetch --db ~/.cargo/advisory-db            # human-readable
cargo audit --no-fetch --db ~/.cargo/advisory-db --json     # machine-readable

# Reverse dependency paths for an advisory, straight from Cargo.lock
cargo tree -i quick-xml@0.38.4

# Full policy check (advisories + licenses + bans), as CI runs it
cargo deny --all-features check
```

Test-function count quoted in §3:

```bash
grep -rn '#\[tokio::test\]\|#\[test\]' --include=*.rs crates | wc -l   # 390
```
