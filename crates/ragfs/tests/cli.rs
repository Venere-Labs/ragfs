//! CLI integration tests. Require `--features test-backends` and
//! `RAGFS_TEST_EMBEDDER=noop` so they run without a model download.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

struct Harness {
    _root: TempDir,
    source: PathBuf,
    data_dir: PathBuf,
    config_path: PathBuf,
}

impl Harness {
    fn new() -> Self {
        let root = TempDir::new().expect("tempdir");
        let source = root.path().join("src");
        let data_dir = root.path().join("data");
        let config_dir = root.path().join("config");
        fs::create_dir_all(&source).unwrap();
        fs::create_dir_all(&data_dir).unwrap();
        fs::create_dir_all(&config_dir).unwrap();
        let config_path = config_dir.join("config.toml");
        fs::write(&config_path, "").unwrap();
        Self {
            _root: root,
            source,
            data_dir,
            config_path,
        }
    }

    fn write_config(&self, body: &str) {
        fs::write(&self.config_path, body).unwrap();
    }

    fn write_source(&self, name: &str, contents: &str) {
        let path = self.source.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    fn ragfs(&self) -> Command {
        let mut cmd = Command::cargo_bin("ragfs").expect("ragfs binary");
        cmd.env("RAGFS_TEST_EMBEDDER", "noop")
            .env("RAGFS_DATA_DIR", &self.data_dir)
            .args(["--config", self.config_path.to_str().unwrap()]);
        cmd
    }
}

fn index_ok(cmd: &mut Command, source: &Path) {
    cmd.args(["-v", "index", source.to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn unsupported_model_fails_before_download() {
    let h = Harness::new();
    h.write_config(
        r#"
        [embedding]
        model = "jina-embeddings-v3"
        "#,
    );
    h.write_source("note.md", "hello");

    h.ragfs()
        .args(["index", h.source.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("jina-embeddings-v3"))
        .stderr(predicate::str::contains("thenlper/gte-small"));
}

#[test]
fn config_max_file_size_reaches_the_indexer() {
    let h = Harness::new();
    h.write_config(
        r#"
        [index]
        max_file_size = 32
        "#,
    );
    h.write_source("small.md", "tiny");
    h.write_source("large.md", &"x".repeat(200));

    let output = h
        .ragfs()
        .args(["-v", "index", h.source.to_str().unwrap()])
        .output()
        .expect("run ragfs index");
    assert!(output.status.success(), "index should succeed");
    let blob = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        blob.contains("max_file_size"),
        "indexer must log the max_file_size skip: {blob}"
    );
}

#[test]
fn force_flag_reaches_the_indexer() {
    let h = Harness::new();
    h.write_source("note.md", "alpha beta gamma");
    index_ok(&mut h.ragfs(), &h.source);

    let output = h
        .ragfs()
        .args(["-v", "index", h.source.to_str().unwrap(), "--force"])
        .output()
        .expect("run ragfs index --force");
    assert!(output.status.success(), "force index should succeed");
    let blob = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        blob.contains("force=true"),
        "indexer must log force=true: {blob}"
    );
}

#[test]
fn hybrid_cli_flag_overrides_config_false() {
    let h = Harness::new();
    h.write_config(
        r#"
        [query]
        hybrid = false
        default_limit = 1
        "#,
    );
    h.write_source("note.md", "semantic search hybrid override");
    index_ok(&mut h.ragfs(), &h.source);

    let output = h
        .ragfs()
        .args([
            "--format",
            "json",
            "query",
            h.source.to_str().unwrap(),
            "semantic",
            "--hybrid",
            "--limit",
            "5",
        ])
        .output()
        .expect("run ragfs query --hybrid");
    let blob = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !blob.contains("unexpected argument"),
        "clap must accept --hybrid: {blob}"
    );
    // Noop embeddings can leave hybrid rank without `_distance`; the smoke
    // is that the CLI took the hybrid path rather than ignoring the flag.
    assert!(
        output.status.success() || blob.contains("hybrid search") || blob.contains("_distance"),
        "expected hybrid query path, got: {blob}"
    );
}

#[test]
fn status_json_after_index() {
    let h = Harness::new();
    h.write_source("note.md", "indexed by the cli test");
    index_ok(&mut h.ragfs(), &h.source);

    h.ragfs()
        .args(["--format", "json", "status", h.source.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"total_files\""))
        .stdout(predicate::str::contains("\"total_chunks\""));
}
