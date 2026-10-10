# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in RAGFS, please report it responsibly.

### How to Report

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. Prefer [GitHub private vulnerability reporting](https://github.com/Venere-Labs/ragfs/security/advisories/new)
3. Fallback email: info@gianlucamazza.it
4. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Next minor release

## Security Measures

### Local-First Architecture

RAGFS operates entirely locally by default:
- No network connections required after initial model download
- All data stored locally in `~/.local/share/ragfs/`
- No telemetry or analytics

### Safe File Operations

- **Soft Delete**: Files are moved to trash, not permanently deleted
- **Audit History**: All operations are logged with undo capability
- **Atomic Operations**: Batch operations are all-or-nothing

### Code Security

- Dependencies checked via `cargo deny` / `cargo audit` in CI
- Public traits avoid `unsafe`; the embedder uses `unsafe` mmap for safetensors
- Content-addressed storage using blake3 hashes
- Path validation (`ragfs_core::resolve_under_root`) is enforced on FUSE ops/semantic writes and on `SafetyManager` soft-delete / restore / undo

### FUSE / agent surface

- `.ops/`, `.safety/`, and `.semantic/` can mutate the source tree
- `allow_other` exposes that surface to every user on the machine — do not enable it on shared hosts without a policy
- Soft-delete can be bypassed if safety is disabled
- Index chunks store file text under `~/.local/share/ragfs/` (protect that directory)

### FUSE Security

When running as a FUSE filesystem:
- Operates with user permissions (no elevated privileges required)
- Mount points are user-owned
- No root access required for standard operations

## Security Best Practices

When using RAGFS:

1. **Keep Updated**: Always use the latest version
2. **Limit Access**: Only mount filesystems with appropriate permissions
3. **Review Plans**: Always review organization/cleanup plans before approving
4. **Backup Data**: Maintain regular backups of important files
5. **Secure Indices**: Protect the `~/.local/share/ragfs/` directory

## Advisory response

CI runs `cargo deny check` and `cargo audit` on every PR and on a schedule
(`security.yml`). `unmaintained` is informational for transitive crates
(`deny.toml`: `unmaintained = "workspace"`). There is **no** `ignore:` list
today (`ignore = []` in `deny.toml` and in the workflows).

### Severity ladder

| Kind (RustSec) | Merge rule | Tracker |
|---|---|---|
| **vulnerability** | Blocker. Do not merge until the ID is gone from the default-feature graph, or the change that introduced it is reverted. | `security` issue, P0 |
| **unsound** | Fix in this repo, **or** write a reachability justification in `docs/AUDIT.md` §1.2 with a review date. Unjustified unsound is a blocker. | `security` issue |
| **unmaintained** | Not a merge blocker. Open or reuse a `security` issue and keep a row in `docs/AUDIT.md` §1.2 naming the crate, who pulls it, and what would have to change upstream. | `security` + `backlog` |
| **yanked** | Warn. Prefer `cargo update -p <crate>` when a replacement exists; otherwise treat like unmaintained. | comment on the PR |

A GitHub Dependabot **alert** is triaged with the same ladder using the
advisory ID, not the Dependabot severity label.

### `ignore:` list

- **Owner:** Lab (Venere Labs maintainers of this repository).
- **Default:** `ignore = []`. An empty list is a feature, not an omission.
- **Adding an ID** requires all of: (1) a reachability analysis (the crate is
  not in the default-feature graph, or the vulnerable function is unreachable),
  (2) a row in `docs/AUDIT.md` §1.2 with the ID, path, and a review date,
  (3) a PR that changes `deny.toml` / workflow `ignore` **and** AUDIT.md in the
  same commit. Expiry of the review date re-opens the ID as a vulnerability
  blocker.

### Cadence

Re-read `docs/AUDIT.md` §1.2 at every minor release, and in any case no later
than **2026-12-31**. The date in §1.3 is the source of truth; move it forward
when the review happens.

### Issues opened by `rustsec/audit-check`

1. Label `security`. Add `rust` or `python` to match the ecosystem.
2. Search for the same advisory ID. Close duplicates (`#81` of `#82` is the
   model: same RUSTSEC, different versions).
3. If the crate is only pulled by a feature or PR already in flight, comment
   `blocked-by #<n>` and leave the issue open until that PR lands.
4. If the ID is already a §1.2 row (accepted unmaintained), label `backlog`
   and point at the row. Do not re-open a second issue.

## Third-Party Dependencies

We regularly audit our dependencies:

- Rust: `cargo deny` / `cargo audit` on every PR and on a schedule (see ladder above)
- Python: Security scanners in CI/CD
- Workspace crate versions are pinned; some transitive crates may have multiple versions (see `deny.toml`)

## Acknowledgments

We appreciate responsible security researchers who help keep RAGFS safe.
Contributors will be acknowledged (with permission) in release notes.
