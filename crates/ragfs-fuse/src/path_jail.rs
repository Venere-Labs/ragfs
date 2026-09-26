//! Path jail: keep agent-facing paths inside a source root.
//!
//! Absolute paths, `..` traversal, and symlink hops that leave the root are rejected.

use std::path::{Component, Path, PathBuf};

/// Resolve `path` so the result is inside `root`.
///
/// Relative paths are joined to `root`. Absolute paths are accepted only when
/// they (after normalization and symlink resolution) stay under `root`.
pub fn resolve_under_root(root: &Path, path: &Path) -> Result<PathBuf, String> {
    let root = absolute_root(root)?;
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };

    let normalized = normalize_lexically(&joined);
    let resolved = resolve_existing_prefix(&normalized)?;

    if !resolved.starts_with(&root) {
        return Err(format!("Path escapes source root: {}", path.display()));
    }

    Ok(resolved)
}

fn absolute_root(root: &Path) -> Result<PathBuf, String> {
    if let Ok(canon) = root.canonicalize() {
        return Ok(canon);
    }

    let abs = if root.is_absolute() {
        normalize_lexically(root)
    } else {
        let cwd = std::env::current_dir().map_err(|e| format!("Invalid source root: {e}"))?;
        normalize_lexically(&cwd.join(root))
    };
    Ok(abs)
}

fn normalize_lexically(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => out.push(prefix.as_os_str()),
            Component::RootDir => out.push(component),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(part) => out.push(part),
        }
    }
    out
}

/// Canonicalize the longest existing prefix so symlink hops are visible.
fn resolve_existing_prefix(path: &Path) -> Result<PathBuf, String> {
    let mut current = path.to_path_buf();
    let mut missing = Vec::new();

    while !current.as_os_str().is_empty() && !current.exists() {
        match current.file_name() {
            Some(name) => {
                missing.push(name.to_os_string());
                match current.parent() {
                    Some(parent) if parent != current => current = parent.to_path_buf(),
                    _ => break,
                }
            }
            None => break,
        }
    }

    let mut resolved = if current.exists() {
        current
            .canonicalize()
            .map_err(|e| format!("Failed to resolve path: {e}"))?
    } else {
        current
    };

    for part in missing.into_iter().rev() {
        resolved.push(part);
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn relative_path_stays_inside_root() {
        let root = TempDir::new().unwrap();
        let resolved = resolve_under_root(root.path(), Path::new("a/b.txt")).unwrap();
        assert_eq!(
            resolved,
            root.path().canonicalize().unwrap().join("a/b.txt")
        );
    }

    #[test]
    fn parent_dir_inside_root_is_allowed() {
        let root = TempDir::new().unwrap();
        fs::create_dir(root.path().join("sub")).unwrap();
        let resolved = resolve_under_root(root.path(), Path::new("sub/../kept.txt")).unwrap();
        assert_eq!(
            resolved,
            root.path().canonicalize().unwrap().join("kept.txt")
        );
    }

    #[test]
    fn parent_dir_escape_is_rejected() {
        let root = TempDir::new().unwrap();
        let err = resolve_under_root(root.path(), Path::new("../escape.txt")).unwrap_err();
        assert!(err.contains("escapes"), "{err}");
    }

    #[test]
    fn nested_parent_escape_is_rejected() {
        let root = TempDir::new().unwrap();
        let err = resolve_under_root(root.path(), Path::new("sub/../../escape.txt")).unwrap_err();
        assert!(err.contains("escapes"), "{err}");
    }

    #[test]
    fn absolute_path_inside_root_is_allowed() {
        let root = TempDir::new().unwrap();
        let inside = root.path().join("inside.txt");
        let resolved = resolve_under_root(root.path(), &inside).unwrap();
        assert_eq!(
            resolved,
            root.path().canonicalize().unwrap().join("inside.txt")
        );
    }

    #[test]
    fn absolute_path_outside_root_is_rejected() {
        let root = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let err = resolve_under_root(root.path(), &outside.path().join("x.txt")).unwrap_err();
        assert!(err.contains("escapes"), "{err}");
    }

    #[test]
    #[cfg(unix)]
    fn symlink_escape_is_rejected() {
        let root = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        std::os::unix::fs::symlink(outside.path(), root.path().join("out")).unwrap();
        let err = resolve_under_root(root.path(), Path::new("out/secret.txt")).unwrap_err();
        assert!(err.contains("escapes"), "{err}");
    }
}
