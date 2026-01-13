#!/bin/bash
set -e

DOCS_PATH="${DOCUMENTS_PATH:-/data/docs}"
DB_PATH="${RAGFS_DB_PATH:-/data/index}"

echo "=== RAGFS Indexer (Watch Mode) ==="
echo "Documents path: $DOCS_PATH"
echo "Database path: $DB_PATH"

# Create docs directory if it doesn't exist
mkdir -p "$DOCS_PATH"

# Check if documents exist
DOC_COUNT=$(find "$DOCS_PATH" -type f 2>/dev/null | wc -l)
if [ "$DOC_COUNT" -eq 0 ]; then
    echo "Warning: No documents found in $DOCS_PATH"
    echo "Add documents via upload or to the sample-docs directory."
    echo "The indexer will automatically detect and index new files."
fi

echo "Found $DOC_COUNT files to index"
echo "Starting indexer in watch mode..."
echo "New files will be automatically indexed when added."
echo ""

# Run indexer in watch mode - this will:
# 1. Index all existing files
# 2. Watch for new/modified/deleted files
# 3. Automatically re-index changes
exec ragfs -v index "$DOCS_PATH" --db "$DB_PATH" --watch
