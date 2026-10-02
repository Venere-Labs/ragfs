//! Library surface for the RAGFS CLI.
//!
//! The binary (`main.rs`) uses these types to load `config.toml` and apply it
//! to the indexer, query executor, and embedder.

pub mod config;

pub use config::{Config, ConfigError, SUPPORTED_EMBEDDING_MODEL, resolve_supported_model};

#[cfg(test)]
mod docs_drift {
    /// Fail if `CLAUDE.md` / `AGENTS.md` omit a Cargo workspace member.
    /// Do not assert a test-function count; that number drifts every PR.
    #[test]
    fn agent_docs_list_every_workspace_member() {
        let manifest = include_str!("../../../Cargo.toml");
        let claude = include_str!("../../../CLAUDE.md");
        let agents = include_str!("../../../AGENTS.md");
        let members = workspace_members(manifest);
        assert!(
            !members.is_empty(),
            "failed to parse [workspace] members from Cargo.toml"
        );
        for member in members {
            let name = member.rsplit('/').next().expect("crate path");
            let needle = format!("`{name}`");
            assert!(
                claude.contains(&needle),
                "CLAUDE.md does not mention workspace member {name}"
            );
            assert!(
                agents.contains(&needle),
                "AGENTS.md does not mention workspace member {name}"
            );
        }
    }

    fn workspace_members(manifest: &str) -> Vec<&str> {
        let start = manifest.find("members = [").expect("members = [");
        let rest = &manifest[start..];
        let end = rest.find(']').expect("members ]");
        rest[..end]
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                let line = line.strip_prefix('"')?;
                let line = line.trim_end_matches(',');
                let line = line.strip_suffix('"')?;
                Some(line)
            })
            .collect()
    }
}
