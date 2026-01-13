# RAGFS Docker RAG Stack v2

A complete, containerized RAG (Retrieval Augmented Generation) stack with full file management capabilities.

## Features

- **Semantic Search** - Find documents by meaning, not just keywords
- **File Upload** - Upload documents directly from the UI
- **Real-Time Indexing** - File watcher automatically indexes new files
- **Safety Layer** - Soft delete with trash, restore, and undo
- **AI Organization** - Find duplicates, cleanup candidates, auto-organize
- **Local & Private** - All processing happens on your machine

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                           │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │   ollama     │  │ ragfs-indexer│  │   rag-app    │            │
│  │  (llama3.2)  │  │  (--watch)   │  │  (Chainlit)  │            │
│  └──────────────┘  └──────┬───────┘  └──────┬───────┘            │
│                           │                 │                     │
│                    ┌──────▼─────────────────▼──────┐              │
│                    │       Shared Volumes          │              │
│                    │  - documents (uploads)        │              │
│                    │  - ragfs-index (LanceDB)      │              │
│                    │  - ragfs-trash (soft deletes) │              │
│                    └───────────────────────────────┘              │
│                                                                   │
│  UI Features:                                                     │
│  ├── Chat        → RAG query with streaming                      │
│  ├── Upload      → Add files, auto-indexed                       │
│  ├── /files      → Browse documents                              │
│  ├── /trash      → View/restore deleted                          │
│  ├── /history    → Operation log with undo                       │
│  ├── /duplicates → Find duplicate files                          │
│  └── /organize   → AI organization plans                         │
└──────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Add Documents

Place your documents in the `sample-docs/` directory:

```bash
cp -r /path/to/your/docs ./sample-docs/
```

Or upload files directly through the web UI after starting.

### 2. Start the Stack

```bash
# Copy environment file (optional)
cp .env.example .env

# Start all services
docker compose up --build
```

First run downloads:
- Embedding model (~100MB)
- Ollama model (~2GB for llama3.2)

### 3. Open the UI

Navigate to [http://localhost:8000](http://localhost:8000)

## Usage

### Semantic Search

Just type your question:
- "How does authentication work?"
- "Find API documentation"
- "What are the main components?"

### File Commands

| Command | Description |
|---------|-------------|
| `/files` | List all indexed documents |
| `/trash` | View deleted files (can restore) |
| `/history` | Operation history (can undo) |

### AI Commands

| Command | Description |
|---------|-------------|
| `/duplicates` | Find duplicate files |
| `/cleanup` | Analyze cleanup candidates |
| `/organize` | Create AI organization plan |
| `/organize by_type` | Organize by file type |
| `/organize by_date` | Organize by date |
| `/pending` | View pending plans |

### File Operations

- **Upload**: Use the attachment button in the chat
- **Delete**: Click delete button next to a file in `/files`
- **Restore**: Click restore button in `/trash`
- **Undo**: Click undo button in `/history`

### Organization Workflow

1. Run `/organize` to create a plan
2. Review the proposed actions
3. Click **Approve** to execute or **Reject** to discard

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_MODEL` | Ollama model to use | `llama3.2` |
| `RAGFS_DB_PATH` | Index database path | `/data/index` |
| `DOCUMENTS_PATH` | Documents directory | `/data/docs` |
| `RAGFS_TRASH_PATH` | Trash directory | `/data/trash` |

### GPU Support (NVIDIA)

Uncomment the GPU section in `docker-compose.yml`:

```yaml
ollama:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Development Mode

For hot-reload during development:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

## File Structure

```
docker-rag-stack/
├── docker-compose.yml       # Main orchestration
├── docker-compose.dev.yml   # Development overrides
├── .env.example             # Environment template
├── README.md                # This file
│
├── ragfs-indexer/
│   ├── Dockerfile           # Multi-stage Rust build
│   └── entrypoint.sh        # Watch mode startup
│
├── rag-app/
│   ├── Dockerfile           # Python + Chainlit
│   ├── requirements.txt     # Python dependencies
│   ├── app.py               # Main Chainlit app
│   ├── rag_chain.py         # LangChain RAG pipeline
│   ├── file_manager.py      # File operations + safety
│   ├── organizer.py         # AI organization
│   ├── components.py        # UI components
│   └── chainlit.md          # Welcome message
│
└── sample-docs/             # Documents to index
    └── README.md            # Sample documentation
```

## Troubleshooting

### Build fails

Run from the repository root:

```bash
cd /path/to/ragfs
docker compose -f examples/docker-rag-stack/docker-compose.yml up --build
```

### Files not indexed

Check indexer logs:

```bash
docker compose logs ragfs-indexer
```

New files are auto-indexed within seconds (500ms debounce).

### Ollama model slow

First download takes a few minutes. Check progress:

```bash
docker compose logs model-puller
```

### Out of memory

Use a smaller model:

```bash
OLLAMA_MODEL=phi3 docker compose up
```

## Supported File Types

- **Text**: .txt, .md, .rst, .html
- **Code**: .py, .rs, .js, .ts, .go, .java, .c, .cpp (40+ languages)
- **Data**: .json, .yaml, .yml, .toml, .xml, .csv
- **Documents**: .pdf, .doc, .docx

## Resources

- [RAGFS Documentation](https://github.com/Venere-Labs/ragfs)
- [Chainlit Documentation](https://docs.chainlit.io)
- [Ollama Models](https://ollama.ai/library)
- [LangChain Documentation](https://python.langchain.com)

## License

MIT OR Apache-2.0
