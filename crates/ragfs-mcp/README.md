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

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "ragfs": {
      "command": "ragfs-mcp"
    }
  }
}
```

## Documentation

See [MCP.md](../../docs/MCP.md) for full documentation.

## License

MIT OR Apache-2.0
