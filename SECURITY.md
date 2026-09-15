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
2. Email security concerns to: security@venere-labs.com
3. Include:
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
- Path validation is enforced at the FUSE ops/semantic boundary when path jail is enabled

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

## Third-Party Dependencies

We regularly audit our dependencies:

- Rust: `cargo audit` run on every release
- Python: Security scanners in CI/CD
- Workspace crate versions are pinned; some transitive crates may have multiple versions (see `deny.toml`)

## Acknowledgments

We appreciate responsible security researchers who help keep RAGFS safe.
Contributors will be acknowledged (with permission) in release notes.
