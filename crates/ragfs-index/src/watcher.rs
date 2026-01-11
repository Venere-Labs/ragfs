//! File system watcher for detecting changes.

use notify_debouncer_full::notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_full::{new_debouncer, DebounceEventResult, Debouncer, RecommendedCache};
use ragfs_core::FileEvent;
use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;
use tokio::sync::mpsc as tokio_mpsc;
use tracing::{debug, error, warn};

/// File system watcher with debouncing.
pub struct FileWatcher {
    debouncer: Debouncer<RecommendedWatcher, RecommendedCache>,
}

impl FileWatcher {
    /// Create a new file watcher.
    pub fn new(
        event_tx: tokio_mpsc::Sender<FileEvent>,
        debounce_duration: Duration,
    ) -> Result<Self, notify::Error> {
        let (tx, rx) = mpsc::channel();

        // Spawn thread to convert events
        let event_tx_clone = event_tx.clone();
        std::thread::spawn(move || {
            while let Ok(result) = rx.recv() {
                if let Err(e) = handle_debounced_events(result, &event_tx_clone) {
                    error!("Error handling file events: {e}");
                }
            }
        });

        let debouncer = new_debouncer(debounce_duration, None, move |result| {
            let _ = tx.send(result);
        })?;

        Ok(Self { debouncer })
    }

    /// Start watching a path.
    pub fn watch(&mut self, path: &Path) -> Result<(), notify_debouncer_full::notify::Error> {
        debug!("Starting to watch: {:?}", path);
        self.debouncer.watch(path, RecursiveMode::Recursive)?;
        Ok(())
    }

    /// Stop watching a path.
    pub fn unwatch(&mut self, path: &Path) -> Result<(), notify_debouncer_full::notify::Error> {
        debug!("Stopping watch: {:?}", path);
        self.debouncer.unwatch(path)?;
        Ok(())
    }
}

fn handle_debounced_events(
    result: DebounceEventResult,
    event_tx: &tokio_mpsc::Sender<FileEvent>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match result {
        Ok(events) => {
            for event in events {
                if let Some(file_event) = convert_event(&event) {
                    // Use blocking send since we're in a std thread
                    if event_tx.blocking_send(file_event).is_err() {
                        warn!("Event channel closed");
                        break;
                    }
                }
            }
        }
        Err(errors) => {
            for error in errors {
                error!("Watch error: {error}");
            }
        }
    }
    Ok(())
}

fn convert_event(event: &notify_debouncer_full::DebouncedEvent) -> Option<FileEvent> {
    use notify_debouncer_full::notify::EventKind;

    let path = event.paths.first()?.clone();

    // Skip hidden files and directories
    if path
        .file_name()
        .is_some_and(|name| name.to_string_lossy().starts_with('.'))
    {
        return None;
    }

    match &event.kind {
        EventKind::Create(_) => Some(FileEvent::Created(path)),
        EventKind::Modify(_) => Some(FileEvent::Modified(path)),
        EventKind::Remove(_) => Some(FileEvent::Deleted(path)),
        EventKind::Other => {
            // Handle rename as "other" event with two paths
            if event.paths.len() >= 2 {
                Some(FileEvent::Renamed {
                    from: event.paths[0].clone(),
                    to: event.paths[1].clone(),
                })
            } else {
                None
            }
        }
        _ => None,
    }
}
