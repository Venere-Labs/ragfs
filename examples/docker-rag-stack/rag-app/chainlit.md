# RAGFS File Manager

Welcome to **RAGFS** - your AI-powered semantic document manager!

## Features

### Semantic Search
Ask questions in natural language to find relevant documents:
- "How does authentication work?"
- "Find documentation about the API"
- "What are the main components?"

### File Management
Manage your documents with full safety features:
- **Upload** files using the attachment button
- **Delete** files safely (moved to trash)
- **Restore** files from trash
- **Undo** any operation

### AI Organization
Let AI help organize your files:
- Find **duplicate** files automatically
- Identify **cleanup** candidates
- Create **organization** plans by topic, type, or date

## Commands

| Command | Description |
|---------|-------------|
| `/files` | List all indexed documents |
| `/trash` | View deleted files |
| `/history` | See operation history |
| `/duplicates` | Find duplicate files |
| `/cleanup` | Analyze cleanup candidates |
| `/organize` | Create AI organization plan |
| `/pending` | View pending plans |
| `/help` | Show all commands |

## How It Works

1. **Upload** documents or add them to the `sample-docs` folder
2. Files are **automatically indexed** using semantic embeddings
3. **Search** by meaning, not just keywords
4. **Manage** files with full trash/undo support
5. **Organize** with AI-powered suggestions

## Safety First

- All deletions are **soft deletes** (recoverable from trash)
- Full **operation history** with undo support
- Organization plans require **explicit approval**
- No data is sent to external servers

---

*Powered by [RAGFS](https://github.com/Venere-Labs/ragfs), [Chainlit](https://chainlit.io), and [Ollama](https://ollama.ai)*
