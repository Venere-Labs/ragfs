# RAGFS MCP Server

MCP (Model Context Protocol) server that exposes RAGFS semantic filesystem capabilities to AI assistants like Claude.

## Installation

```bash
pip install ragfs-mcp
```

## Usage

```bash
# Run the server
ragfs-mcp

# Or as a Python module
python -m ragfs_mcp
```

## Claude Desktop Configuration

Add to your Claude Desktop config. Set `RAGFS_SOURCE_PATH` to the same directory you indexed with `ragfs index` — MCP hashes that canonical path (blake3, first 16 hex chars) and opens `~/.local/share/ragfs/indices/{16hex}/index.lance`, matching the CLI.

```json
{
  "mcpServers": {
    "ragfs": {
      "command": "ragfs-mcp",
      "env": {
        "RAGFS_SOURCE_PATH": "/path/to/project"
      }
    }
  }
}
```

## Documentation

See [MCP.md](../../docs/MCP.md) for full documentation.

## License

MIT OR Apache-2.0
