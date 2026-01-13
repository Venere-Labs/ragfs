//! File system watcher for detecting changes.

use notify_debouncer_full::notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_full::{DebounceEventResult, Debouncer, RecommendedCache, new_debouncer};
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

#[cfg(test)]
mod tests {
    use super::*;
    use notify_debouncer_full::DebouncedEvent;
    use notify_debouncer_full::notify::EventKind;
    use notify_debouncer_full::notify::event::{CreateKind, ModifyKind, RemoveKind};
    use std::path::PathBuf;
    use std::time::Instant;

    fn make_event(kind: EventKind, paths: Vec<PathBuf>) -> DebouncedEvent {
        DebouncedEvent {
            event: notify_debouncer_full::notify::Event {
                kind,
                paths,
                attrs: Default::default(),
            },
            time: Instant::now(),
        }
    }

    #[test]
    fn test_convert_event_create() {
        let path = PathBuf::from("/tmp/test.txt");
        let event = make_event(EventKind::Create(CreateKind::File), vec![path.clone()]);

        let result = convert_event(&event);
        assert!(matches!(result, Some(FileEvent::Created(p)) if p == path));
    }

    #[test]
    fn test_convert_event_modify() {
        use notify_debouncer_full::notify::event::DataChange;
        let path = PathBuf::from("/tmp/test.txt");
        let event = make_event(
            EventKind::Modify(ModifyKind::Data(DataChange::Any)),
            vec![path.clone()],
        );

        let result = convert_event(&event);
        assert!(matches!(result, Some(FileEvent::Modified(p)) if p == path));
    }

    #[test]
    fn test_convert_event_delete() {
        let path = PathBuf::from("/tmp/test.txt");
        let event = make_event(EventKind::Remove(RemoveKind::File), vec![path.clone()]);

        let result = convert_event(&event);
        assert!(matches!(result, Some(FileEvent::Deleted(p)) if p == path));
    }

    #[test]
    fn test_hidden_files_skipped() {
        let path = PathBuf::from("/tmp/.hidden");
        let event = make_event(EventKind::Create(CreateKind::File), vec![path]);

        let result = convert_event(&event);
        assert!(result.is_none());
    }
}
