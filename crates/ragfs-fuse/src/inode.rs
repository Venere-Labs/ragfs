//! Inode management for virtual and real files.

use std::collections::HashMap;
use std::path::PathBuf;

/// Reserved inode numbers.
pub const ROOT_INO: u64 = 1;
pub const RAGFS_DIR_INO: u64 = 2;
pub const QUERY_DIR_INO: u64 = 3;
pub const SEARCH_DIR_INO: u64 = 4;
pub const INDEX_FILE_INO: u64 = 5;
pub const CONFIG_FILE_INO: u64 = 6;
pub const REINDEX_FILE_INO: u64 = 7;
pub const SIMILAR_DIR_INO: u64 = 8;
pub const FIRST_REAL_INO: u64 = 1000;

/// Type of inode.
#[derive(Debug, Clone)]
pub enum InodeKind {
    /// Root of mounted filesystem
    Root,
    /// Virtual .ragfs control directory
    RagfsDir,
    /// Virtual .query directory
    QueryDir,
    /// Dynamic query result file
    QueryResult { query: String },
    /// Virtual .search directory
    SearchDir,
    /// Search results as symlink directory
    SearchResult { query: String },
    /// .index status file
    IndexStatus,
    /// .config file
    Config,
    /// .reindex trigger file
    Reindex,
    /// .similar directory
    SimilarDir,
    /// Similar file lookup
    SimilarLookup { source_path: PathBuf },
    /// Real file/directory passthrough
    Real {
        path: PathBuf,
        underlying_ino: u64,
    },
}

/// Entry in the inode table.
#[derive(Debug, Clone)]
pub struct InodeEntry {
    /// Inode number
    pub ino: u64,
    /// Type of inode
    pub kind: InodeKind,
    /// Parent inode
    pub parent: u64,
    /// Lookup count (for FUSE reference counting)
    pub lookup_count: u64,
}

/// Inode table managing virtual and real file mappings.
pub struct InodeTable {
    /// Inode number -> entry
    inodes: HashMap<u64, InodeEntry>,
    /// Path -> inode for real files
    path_to_ino: HashMap<PathBuf, u64>,
    /// Underlying inode -> our inode (for real files)
    real_ino_map: HashMap<u64, u64>,
    /// Query string -> inode (for query results)
    query_to_ino: HashMap<String, u64>,
    /// Next available inode
    next_ino: u64,
}

impl InodeTable {
    /// Create a new inode table with virtual inodes initialized.
    pub fn new() -> Self {
        let mut table = Self {
            inodes: HashMap::new(),
            path_to_ino: HashMap::new(),
            real_ino_map: HashMap::new(),
            query_to_ino: HashMap::new(),
            next_ino: FIRST_REAL_INO,
        };
        table.init_virtual_inodes();
        table
    }

    fn init_virtual_inodes(&mut self) {
        // Root
        self.inodes.insert(
            ROOT_INO,
            InodeEntry {
                ino: ROOT_INO,
                kind: InodeKind::Root,
                parent: ROOT_INO,
                lookup_count: 1,
            },
        );

        // .ragfs directory
        self.inodes.insert(
            RAGFS_DIR_INO,
            InodeEntry {
                ino: RAGFS_DIR_INO,
                kind: InodeKind::RagfsDir,
                parent: ROOT_INO,
                lookup_count: 0,
            },
        );

        // .query directory
        self.inodes.insert(
            QUERY_DIR_INO,
            InodeEntry {
                ino: QUERY_DIR_INO,
                kind: InodeKind::QueryDir,
                parent: RAGFS_DIR_INO,
                lookup_count: 0,
            },
        );

        // .search directory
        self.inodes.insert(
            SEARCH_DIR_INO,
            InodeEntry {
                ino: SEARCH_DIR_INO,
                kind: InodeKind::SearchDir,
                parent: RAGFS_DIR_INO,
                lookup_count: 0,
            },
        );

        // .index file
        self.inodes.insert(
            INDEX_FILE_INO,
            InodeEntry {
                ino: INDEX_FILE_INO,
                kind: InodeKind::IndexStatus,
                parent: RAGFS_DIR_INO,
                lookup_count: 0,
            },
        );

        // .config file
        self.inodes.insert(
            CONFIG_FILE_INO,
            InodeEntry {
                ino: CONFIG_FILE_INO,
                kind: InodeKind::Config,
                parent: RAGFS_DIR_INO,
                lookup_count: 0,
            },
        );

        // .reindex file
        self.inodes.insert(
            REINDEX_FILE_INO,
            InodeEntry {
                ino: REINDEX_FILE_INO,
                kind: InodeKind::Reindex,
                parent: RAGFS_DIR_INO,
                lookup_count: 0,
            },
        );

        // .similar directory
        self.inodes.insert(
            SIMILAR_DIR_INO,
            InodeEntry {
                ino: SIMILAR_DIR_INO,
                kind: InodeKind::SimilarDir,
                parent: RAGFS_DIR_INO,
                lookup_count: 0,
            },
        );
    }

    /// Get an inode entry.
    pub fn get(&self, ino: u64) -> Option<&InodeEntry> {
        self.inodes.get(&ino)
    }

    /// Get or create inode for a real path.
    pub fn get_or_create_real(&mut self, path: PathBuf, underlying_ino: u64) -> u64 {
        if let Some(&ino) = self.path_to_ino.get(&path) {
            return ino;
        }

        let ino = self.next_ino;
        self.next_ino += 1;

        self.inodes.insert(
            ino,
            InodeEntry {
                ino,
                kind: InodeKind::Real {
                    path: path.clone(),
                    underlying_ino,
                },
                parent: ROOT_INO,
                lookup_count: 0,
            },
        );

        self.path_to_ino.insert(path, ino);
        self.real_ino_map.insert(underlying_ino, ino);

        ino
    }

    /// Get or create inode for a query result.
    pub fn get_or_create_query_result(&mut self, parent: u64, query: String) -> u64 {
        if let Some(&ino) = self.query_to_ino.get(&query) {
            return ino;
        }

        let ino = self.next_ino;
        self.next_ino += 1;

        self.inodes.insert(
            ino,
            InodeEntry {
                ino,
                kind: InodeKind::QueryResult { query: query.clone() },
                parent,
                lookup_count: 0,
            },
        );

        self.query_to_ino.insert(query, ino);

        ino
    }

    /// Increment lookup count.
    pub fn lookup(&mut self, ino: u64) {
        if let Some(entry) = self.inodes.get_mut(&ino) {
            entry.lookup_count += 1;
        }
    }

    /// Decrement lookup count.
    pub fn forget(&mut self, ino: u64, nlookup: u64) {
        if let Some(entry) = self.inodes.get_mut(&ino) {
            entry.lookup_count = entry.lookup_count.saturating_sub(nlookup);
        }
    }

    /// Check if inode is virtual (part of .ragfs).
    pub fn is_virtual(&self, ino: u64) -> bool {
        ino < FIRST_REAL_INO
    }

    /// Get inode by path (for real files).
    pub fn get_by_path(&self, path: &PathBuf) -> Option<u64> {
        self.path_to_ino.get(path).copied()
    }
}

impl Default for InodeTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Constants Tests ==========

    #[test]
    fn test_reserved_inode_constants() {
        assert_eq!(ROOT_INO, 1);
        assert_eq!(RAGFS_DIR_INO, 2);
        assert_eq!(QUERY_DIR_INO, 3);
        assert_eq!(SEARCH_DIR_INO, 4);
        assert_eq!(INDEX_FILE_INO, 5);
        assert_eq!(CONFIG_FILE_INO, 6);
        assert_eq!(REINDEX_FILE_INO, 7);
        assert_eq!(SIMILAR_DIR_INO, 8);
        assert_eq!(FIRST_REAL_INO, 1000);
    }

    #[test]
    fn test_reserved_inodes_are_below_first_real() {
        assert!(ROOT_INO < FIRST_REAL_INO);
        assert!(RAGFS_DIR_INO < FIRST_REAL_INO);
        assert!(QUERY_DIR_INO < FIRST_REAL_INO);
        assert!(SEARCH_DIR_INO < FIRST_REAL_INO);
        assert!(INDEX_FILE_INO < FIRST_REAL_INO);
        assert!(CONFIG_FILE_INO < FIRST_REAL_INO);
        assert!(REINDEX_FILE_INO < FIRST_REAL_INO);
        assert!(SIMILAR_DIR_INO < FIRST_REAL_INO);
    }

    // ========== InodeKind Tests ==========

    #[test]
    fn test_inode_kind_root() {
        let kind = InodeKind::Root;
        let debug_str = format!("{:?}", kind);
        assert!(debug_str.contains("Root"));
    }

    #[test]
    fn test_inode_kind_ragfs_dir() {
        let kind = InodeKind::RagfsDir;
        let debug_str = format!("{:?}", kind);
        assert!(debug_str.contains("RagfsDir"));
    }

    #[test]
    fn test_inode_kind_query_result() {
        let kind = InodeKind::QueryResult {
            query: "test query".to_string(),
        };
        let debug_str = format!("{:?}", kind);
        assert!(debug_str.contains("QueryResult"));
        assert!(debug_str.contains("test query"));
    }

    #[test]
    fn test_inode_kind_search_result() {
        let kind = InodeKind::SearchResult {
            query: "find files".to_string(),
        };
        let debug_str = format!("{:?}", kind);
        assert!(debug_str.contains("SearchResult"));
    }

    #[test]
    fn test_inode_kind_real() {
        let kind = InodeKind::Real {
            path: PathBuf::from("/test/file.txt"),
            underlying_ino: 12345,
        };
        let debug_str = format!("{:?}", kind);
        assert!(debug_str.contains("Real"));
        assert!(debug_str.contains("file.txt"));
    }

    #[test]
    fn test_inode_kind_similar_lookup() {
        let kind = InodeKind::SimilarLookup {
            source_path: PathBuf::from("/source/doc.pdf"),
        };
        let debug_str = format!("{:?}", kind);
        assert!(debug_str.contains("SimilarLookup"));
    }

    #[test]
    fn test_inode_kind_clone() {
        let kind = InodeKind::QueryResult {
            query: "cloned query".to_string(),
        };
        let cloned = kind.clone();
        if let InodeKind::QueryResult { query } = cloned {
            assert_eq!(query, "cloned query");
        } else {
            panic!("Clone produced wrong variant");
        }
    }

    // ========== InodeEntry Tests ==========

    #[test]
    fn test_inode_entry_creation() {
        let entry = InodeEntry {
            ino: 100,
            kind: InodeKind::Root,
            parent: 1,
            lookup_count: 5,
        };
        assert_eq!(entry.ino, 100);
        assert_eq!(entry.parent, 1);
        assert_eq!(entry.lookup_count, 5);
    }

    #[test]
    fn test_inode_entry_debug() {
        let entry = InodeEntry {
            ino: 42,
            kind: InodeKind::IndexStatus,
            parent: 2,
            lookup_count: 3,
        };
        let debug_str = format!("{:?}", entry);
        assert!(debug_str.contains("42"));
        assert!(debug_str.contains("IndexStatus"));
    }

    #[test]
    fn test_inode_entry_clone() {
        let entry = InodeEntry {
            ino: 50,
            kind: InodeKind::Config,
            parent: 2,
            lookup_count: 10,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.ino, 50);
        assert_eq!(cloned.lookup_count, 10);
    }

    // ========== InodeTable Creation Tests ==========

    #[test]
    fn test_inode_table_new() {
        let table = InodeTable::new();
        assert!(table.get(ROOT_INO).is_some());
        assert!(table.get(RAGFS_DIR_INO).is_some());
        assert!(table.get(QUERY_DIR_INO).is_some());
    }

    #[test]
    fn test_inode_table_default() {
        let table = InodeTable::default();
        assert!(table.get(ROOT_INO).is_some());
    }

    #[test]
    fn test_virtual_inodes_initialized() {
        let table = InodeTable::new();

        // Root
        let root = table.get(ROOT_INO).expect("Root should exist");
        assert!(matches!(root.kind, InodeKind::Root));
        assert_eq!(root.parent, ROOT_INO);

        // RagfsDir
        let ragfs = table.get(RAGFS_DIR_INO).expect("RagfsDir should exist");
        assert!(matches!(ragfs.kind, InodeKind::RagfsDir));
        assert_eq!(ragfs.parent, ROOT_INO);

        // QueryDir
        let query = table.get(QUERY_DIR_INO).expect("QueryDir should exist");
        assert!(matches!(query.kind, InodeKind::QueryDir));
        assert_eq!(query.parent, RAGFS_DIR_INO);

        // SearchDir
        let search = table.get(SEARCH_DIR_INO).expect("SearchDir should exist");
        assert!(matches!(search.kind, InodeKind::SearchDir));
        assert_eq!(search.parent, RAGFS_DIR_INO);

        // IndexStatus
        let index = table.get(INDEX_FILE_INO).expect("IndexFile should exist");
        assert!(matches!(index.kind, InodeKind::IndexStatus));

        // Config
        let config = table.get(CONFIG_FILE_INO).expect("ConfigFile should exist");
        assert!(matches!(config.kind, InodeKind::Config));

        // Reindex
        let reindex = table.get(REINDEX_FILE_INO).expect("ReindexFile should exist");
        assert!(matches!(reindex.kind, InodeKind::Reindex));

        // SimilarDir
        let similar = table.get(SIMILAR_DIR_INO).expect("SimilarDir should exist");
        assert!(matches!(similar.kind, InodeKind::SimilarDir));
    }

    // ========== get() Tests ==========

    #[test]
    fn test_get_existing_inode() {
        let table = InodeTable::new();
        assert!(table.get(ROOT_INO).is_some());
    }

    #[test]
    fn test_get_nonexistent_inode() {
        let table = InodeTable::new();
        assert!(table.get(99999).is_none());
    }

    // ========== get_or_create_real() Tests ==========

    #[test]
    fn test_get_or_create_real_new_path() {
        let mut table = InodeTable::new();
        let path = PathBuf::from("/test/file.txt");
        let underlying_ino = 12345;

        let ino = table.get_or_create_real(path.clone(), underlying_ino);

        assert!(ino >= FIRST_REAL_INO);
        let entry = table.get(ino).expect("Entry should exist");
        if let InodeKind::Real {
            path: p,
            underlying_ino: u,
        } = &entry.kind
        {
            assert_eq!(p, &path);
            assert_eq!(*u, underlying_ino);
        } else {
            panic!("Expected Real inode kind");
        }
    }

    #[test]
    fn test_get_or_create_real_returns_existing() {
        let mut table = InodeTable::new();
        let path = PathBuf::from("/test/file.txt");
        let underlying_ino = 12345;

        let ino1 = table.get_or_create_real(path.clone(), underlying_ino);
        let ino2 = table.get_or_create_real(path.clone(), underlying_ino);

        assert_eq!(ino1, ino2, "Should return same inode for same path");
    }

    #[test]
    fn test_get_or_create_real_increments_ino() {
        let mut table = InodeTable::new();

        let ino1 = table.get_or_create_real(PathBuf::from("/file1.txt"), 100);
        let ino2 = table.get_or_create_real(PathBuf::from("/file2.txt"), 101);
        let ino3 = table.get_or_create_real(PathBuf::from("/file3.txt"), 102);

        assert!(ino2 > ino1);
        assert!(ino3 > ino2);
    }

    #[test]
    fn test_get_or_create_real_parent_is_root() {
        let mut table = InodeTable::new();
        let ino = table.get_or_create_real(PathBuf::from("/test.txt"), 100);

        let entry = table.get(ino).unwrap();
        assert_eq!(entry.parent, ROOT_INO);
    }

    // ========== get_or_create_query_result() Tests ==========

    #[test]
    fn test_get_or_create_query_result_new() {
        let mut table = InodeTable::new();
        let query = "how to implement auth".to_string();

        let ino = table.get_or_create_query_result(QUERY_DIR_INO, query.clone());

        assert!(ino >= FIRST_REAL_INO);
        let entry = table.get(ino).expect("Entry should exist");
        if let InodeKind::QueryResult { query: q } = &entry.kind {
            assert_eq!(q, &query);
        } else {
            panic!("Expected QueryResult kind");
        }
    }

    #[test]
    fn test_get_or_create_query_result_returns_existing() {
        let mut table = InodeTable::new();
        let query = "test query".to_string();

        let ino1 = table.get_or_create_query_result(QUERY_DIR_INO, query.clone());
        let ino2 = table.get_or_create_query_result(QUERY_DIR_INO, query.clone());

        assert_eq!(ino1, ino2, "Should return same inode for same query");
    }

    #[test]
    fn test_get_or_create_query_result_different_queries() {
        let mut table = InodeTable::new();

        let ino1 = table.get_or_create_query_result(QUERY_DIR_INO, "query1".to_string());
        let ino2 = table.get_or_create_query_result(QUERY_DIR_INO, "query2".to_string());

        assert_ne!(ino1, ino2, "Different queries should have different inodes");
    }

    #[test]
    fn test_get_or_create_query_result_preserves_parent() {
        let mut table = InodeTable::new();
        let ino = table.get_or_create_query_result(QUERY_DIR_INO, "test".to_string());

        let entry = table.get(ino).unwrap();
        assert_eq!(entry.parent, QUERY_DIR_INO);
    }

    // ========== lookup() Tests ==========

    #[test]
    fn test_lookup_increments_count() {
        let mut table = InodeTable::new();

        let initial_count = table.get(ROOT_INO).unwrap().lookup_count;
        table.lookup(ROOT_INO);
        let after_count = table.get(ROOT_INO).unwrap().lookup_count;

        assert_eq!(after_count, initial_count + 1);
    }

    #[test]
    fn test_lookup_multiple_times() {
        let mut table = InodeTable::new();

        let initial = table.get(RAGFS_DIR_INO).unwrap().lookup_count;
        table.lookup(RAGFS_DIR_INO);
        table.lookup(RAGFS_DIR_INO);
        table.lookup(RAGFS_DIR_INO);
        let final_count = table.get(RAGFS_DIR_INO).unwrap().lookup_count;

        assert_eq!(final_count, initial + 3);
    }

    #[test]
    fn test_lookup_nonexistent_does_nothing() {
        let mut table = InodeTable::new();
        table.lookup(99999); // Should not panic
    }

    // ========== forget() Tests ==========

    #[test]
    fn test_forget_decrements_count() {
        let mut table = InodeTable::new();
        table.lookup(ROOT_INO);
        table.lookup(ROOT_INO);

        let before = table.get(ROOT_INO).unwrap().lookup_count;
        table.forget(ROOT_INO, 1);
        let after = table.get(ROOT_INO).unwrap().lookup_count;

        assert_eq!(after, before - 1);
    }

    #[test]
    fn test_forget_saturating_sub() {
        let mut table = InodeTable::new();
        // Root starts with lookup_count = 1
        table.forget(ROOT_INO, 100); // Should saturate to 0

        let count = table.get(ROOT_INO).unwrap().lookup_count;
        assert_eq!(count, 0);
    }

    #[test]
    fn test_forget_nonexistent_does_nothing() {
        let mut table = InodeTable::new();
        table.forget(99999, 5); // Should not panic
    }

    // ========== is_virtual() Tests ==========

    #[test]
    fn test_is_virtual_true_for_reserved() {
        let table = InodeTable::new();
        assert!(table.is_virtual(ROOT_INO));
        assert!(table.is_virtual(RAGFS_DIR_INO));
        assert!(table.is_virtual(QUERY_DIR_INO));
        assert!(table.is_virtual(SEARCH_DIR_INO));
        assert!(table.is_virtual(INDEX_FILE_INO));
        assert!(table.is_virtual(CONFIG_FILE_INO));
        assert!(table.is_virtual(REINDEX_FILE_INO));
        assert!(table.is_virtual(SIMILAR_DIR_INO));
    }

    #[test]
    fn test_is_virtual_false_for_real() {
        let table = InodeTable::new();
        assert!(!table.is_virtual(FIRST_REAL_INO));
        assert!(!table.is_virtual(FIRST_REAL_INO + 1));
        assert!(!table.is_virtual(99999));
    }

    #[test]
    fn test_is_virtual_boundary() {
        let table = InodeTable::new();
        assert!(table.is_virtual(FIRST_REAL_INO - 1));
        assert!(!table.is_virtual(FIRST_REAL_INO));
    }

    // ========== get_by_path() Tests ==========

    #[test]
    fn test_get_by_path_existing() {
        let mut table = InodeTable::new();
        let path = PathBuf::from("/documents/test.txt");
        let ino = table.get_or_create_real(path.clone(), 100);

        let found = table.get_by_path(&path);
        assert_eq!(found, Some(ino));
    }

    #[test]
    fn test_get_by_path_nonexistent() {
        let table = InodeTable::new();
        let path = PathBuf::from("/nonexistent/file.txt");

        assert!(table.get_by_path(&path).is_none());
    }

    #[test]
    fn test_get_by_path_different_paths() {
        let mut table = InodeTable::new();
        let path1 = PathBuf::from("/file1.txt");
        let path2 = PathBuf::from("/file2.txt");

        let ino1 = table.get_or_create_real(path1.clone(), 100);
        let ino2 = table.get_or_create_real(path2.clone(), 101);

        assert_eq!(table.get_by_path(&path1), Some(ino1));
        assert_eq!(table.get_by_path(&path2), Some(ino2));
    }

    // ========== Integration Tests ==========

    #[test]
    fn test_mixed_real_and_query_inodes() {
        let mut table = InodeTable::new();

        let real_ino = table.get_or_create_real(PathBuf::from("/doc.txt"), 100);
        let query_ino = table.get_or_create_query_result(QUERY_DIR_INO, "search".to_string());

        assert_ne!(real_ino, query_ino);
        assert!(table.get(real_ino).is_some());
        assert!(table.get(query_ino).is_some());
    }

    #[test]
    fn test_inode_numbers_are_sequential() {
        let mut table = InodeTable::new();

        let ino1 = table.get_or_create_real(PathBuf::from("/a.txt"), 1);
        let ino2 = table.get_or_create_query_result(QUERY_DIR_INO, "q1".to_string());
        let ino3 = table.get_or_create_real(PathBuf::from("/b.txt"), 2);

        assert_eq!(ino2, ino1 + 1);
        assert_eq!(ino3, ino2 + 1);
    }
}
