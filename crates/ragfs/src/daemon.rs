//! Background-mount daemonization without the unmaintained `daemonize` crate.
//!
//! `ragfs mount` (without `--foreground`) forks before Tokio starts so watcher
//! and runtime threads are created only in the child. PID file and log paths
//! stay at `$XDG_RUNTIME_DIR/ragfs/` (or `~/.cache/ragfs/run/`) and
//! `~/.cache/ragfs/logs/`.
//!
//! Uses `nix` (`fork` + `setsid`, stdio redirect, pid file). rustix 1.x has
//! `setsid`/`chdir`/`umask` but not `fork` outside its unstable `runtime`
//! feature.

#![cfg(unix)]

use anyhow::{Context, Result};
use nix::sys::stat::{Mode, umask};
use nix::unistd::{
    ForkResult, chdir, chown, dup2_stderr, dup2_stdin, dup2_stdout, fork, getgid, getpid, getuid,
    setsid,
};
use std::fs::File;
use std::io::Write;
use std::os::unix::io::AsFd;
use std::path::Path;

/// Options matching the previous `daemonize` 0.5 call site.
pub struct DaemonOptions<'a> {
    pub pid_file: &'a Path,
    pub stdout: File,
    pub stderr: File,
    pub working_directory: &'a Path,
    pub chown_pid_file: bool,
}

/// Double-fork into the background, redirect stdio, and write the pid file.
///
/// Parents exit with status 0. On success this function returns only in the
/// daemon process.
pub fn daemonize(opts: DaemonOptions<'_>) -> Result<()> {
    match {
        // SAFETY: `maybe_daemonize_background_mount` runs in `main` before the
        // Tokio runtime or indexer threads exist, so this process is still
        // single-threaded. Only async-signal-safe work happens before we
        // finish daemon setup in the child.
        #[allow(unsafe_code)]
        unsafe {
            fork()
        }
    }
    .context("Failed to daemonize: first fork")?
    {
        ForkResult::Parent { .. } => std::process::exit(0),
        ForkResult::Child => {}
    }

    setsid().context("Failed to daemonize: setsid")?;

    match {
        // SAFETY: still single-threaded; second fork drops session-leader
        // status so the daemon cannot reacquire a controlling terminal.
        #[allow(unsafe_code)]
        unsafe {
            fork()
        }
    }
    .context("Failed to daemonize: second fork")?
    {
        ForkResult::Parent { .. } => {
            // Avoid running atexit/flush in the intermediate child.
            #[allow(unsafe_code)]
            unsafe {
                nix::libc::_exit(0);
            }
        }
        ForkResult::Child => {}
    }

    umask(Mode::empty());
    chdir(opts.working_directory).context("Failed to daemonize: chdir")?;

    let devnull = File::open("/dev/null").context("Failed to open /dev/null")?;
    dup2_stdin(devnull.as_fd()).context("Failed to daemonize: redirect stdin")?;
    dup2_stdout(opts.stdout.as_fd()).context("Failed to daemonize: redirect stdout")?;
    dup2_stderr(opts.stderr.as_fd()).context("Failed to daemonize: redirect stderr")?;

    write_pid_file(opts.pid_file, opts.chown_pid_file)?;
    Ok(())
}

/// Write the current PID to `path`, optionally chowning to the calling user.
pub fn write_pid_file(path: &Path, chown_to_current_user: bool) -> Result<()> {
    let pid = getpid();
    let mut file = File::create(path)
        .with_context(|| format!("Failed to create PID file {}", path.display()))?;
    writeln!(file, "{}", pid.as_raw())
        .with_context(|| format!("Failed to write PID file {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("Failed to sync PID file {}", path.display()))?;

    if chown_to_current_user {
        chown(path, Some(getuid()), Some(getgid()))
            .with_context(|| format!("Failed to chown PID file {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn write_pid_file_records_current_pid() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("ragfs.pid");
        write_pid_file(&path, true).unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents.trim(), getpid().as_raw().to_string());
    }
}
