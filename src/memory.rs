//! Fitting the model into memory without pushing the rest of the system into
//! swap. On Jetson the GPU shares RAM with the CPU, and GPU memory cannot be
//! swapped out: whatever the engine takes, the desktop loses.

use std::fs;

/// Memory left for the rest of the system once the model is loaded. The
/// engine's file size, counted as its cost, over-covers the runtime's own
/// buffers, so in practice the desktop keeps about 1.5 times this.
const SYSTEM_HEADROOM: u64 = 1 << 30;

/// The KV cache is stored in blocks of this many tokens.
const TOKENS_PER_BLOCK: u64 = 64;

/// Below this, a conversation barely fits one question and its answer.
pub const MIN_KV_CACHE_TOKENS: u32 = 4096;

/// Memory the kernel can hand out without swapping (`MemAvailable`), in bytes.
pub fn available_bytes() -> Option<u64> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo.lines().find(|line| line.starts_with("MemAvailable:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

/// The largest KV cache, up to `requested` tokens, that fits in `available`
/// bytes next to an engine of `engine_bytes`, leaving [`SYSTEM_HEADROOM`].
pub fn kv_cache_tokens_that_fit(requested: u32, available: u64, engine_bytes: u64, bytes_per_token: u64) -> u32 {
    let budget = available.saturating_sub(engine_bytes + SYSTEM_HEADROOM);
    let tokens = budget / bytes_per_token.max(1) / TOKENS_PER_BLOCK * TOKENS_PER_BLOCK;
    tokens.min(u64::from(requested)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    /// Qwen2.5-7B: 28 layers × 4 KV heads × 128 dims × (K and V) × fp16.
    const QWEN_7B: u64 = 28 * 4 * 128 * 2 * 2;

    #[test]
    fn plenty_of_memory_gives_the_requested_size() {
        assert_eq!(kv_cache_tokens_that_fit(32768, 9 * GIB, 5 * GIB, QWEN_7B), 32768);
    }

    #[test]
    fn short_memory_shrinks_the_cache_to_whole_blocks() {
        let tokens = kv_cache_tokens_that_fit(32768, 7 * GIB, 5 * GIB, QWEN_7B);
        assert_eq!(tokens, 18688); // 1 GiB / 56 KiB per token, rounded down to 64
        assert_eq!(tokens % 64, 0);
    }

    #[test]
    fn no_room_gives_zero() {
        assert_eq!(kv_cache_tokens_that_fit(32768, 5 * GIB, 5 * GIB, QWEN_7B), 0);
    }
}
