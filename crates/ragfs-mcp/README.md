# RAGFS MCP Server

MCP (Model Context Protocol) server that exposes RAGFS semantic filesystem capabilities to AI assistants like Claude.

## Installation

`ragfs-mcp` is **not yet published on PyPI**. Do not install a package with
this name from PyPI — the name is unclaimed and could be registered by anyone.

Install from source. The server depends on the `ragfs` bindings, which must
also be installed from this repository (not from PyPI):

```bash
# From a clone of this repository
pip install ./crates/ragfs-python
pip install ./crates/ragfs-mcp
# or
pip install -e ./crates/ragfs-mcp

# Without cloning
pip install 'git+https://github.com/Venere-Labs/ragfs#subdirectory=crates/ragfs-python'
pip install 'git+https://github.com/Venere-Labs/ragfs#subdirectory=crates/ragfs-mcp'
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
