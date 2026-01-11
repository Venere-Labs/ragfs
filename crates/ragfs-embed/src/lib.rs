//! Local embedding generation for RAGFS using Candle.

pub mod cache;
pub mod candle;
pub mod pool;

pub use cache::EmbeddingCache;
pub use candle::CandleEmbedder;
pub use pool::EmbedderPool;
