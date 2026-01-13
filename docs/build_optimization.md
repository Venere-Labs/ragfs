# Speeding up Rust Builds

We have optimized the build configuration for `ragfs` to improve compile times. However, the most significant improvements require installing additional system tools.

## 1. Install `mold` (Linker)

`mold` is a modern high-performance linker that can drastically reduce incremental build times (often by 10x or more).

### Installation (Linux)

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install mold clang

# Fedora
sudo dnf install mold clang

# Arch Linux
sudo pacman -S mold clang
```

### Configuration

Once installed, uncomment the corresponding section in `.cargo/config.toml`:

```toml
[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=mold"]
```

## 2. Install `sccache` (Compiler Cache)

`sccache` caches compilation artifacts, significantly speeding up re-compilation of unmodified crates across different workspaces or after clean builds.

### Installation

```bash
cargo install sccache
```

### Configuration

Once installed, uncomment the corresponding section in `.cargo/config.toml`:

```toml
[build]
rustc-wrapper = "sccache"
```

## 3. Optimizations Applied (Out of the Box)

We have already configured:
- **`split-debuginfo = "unpacked"`**: Reduces I/O load during linking on Linux.
- **Dependency Optimization**: Third-party dependencies are compiled with `opt-level = 3` even in dev mode. This slows down the initial clean build slightly but makes the application run much faster during development (crucial for AI/ML libraries like `candle` and `lancedb`).
