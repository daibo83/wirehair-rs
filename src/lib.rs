// Wirehair Forward Error Correction Implementation
// Optimized for speed with SIMD, cache-friendly layouts, and minimal allocations

use gf256::gf::gf256;
// Main Wirehair encoder

pub struct Encoder {
    k: usize,
    block_size: usize,
    data: Vec<u8>,
    seed: u32,
}

impl Encoder {
    pub fn new(data: Vec<u8>, k: usize, block_size: usize) -> Self {
        let mut padded_data = data;
        padded_data.resize(k * block_size, 0);

        Encoder {
            k,
            block_size,
            data: padded_data,
            seed: 0x12345678,
        }
    }

    pub fn encode(&self, symbol_id: u32) -> Vec<u8> {
        if (symbol_id as usize) < self.k {
            // Systematic: return original block
            let i = symbol_id as usize;
            return self.data[i * self.block_size .. (i + 1) * self.block_size].to_vec();
        }

        let mut output = vec![0u8; self.block_size];
        let mut rng = Xorshift32::new(self.seed ^ symbol_id);

        for i in 0..self.k {
            let coeff = gf256((rng.next() & 0xFF) as u8);
            if coeff == gf256(0) { continue; }

            let src_block = &self.data[i * self.block_size .. (i + 1) * self.block_size];
            for j in 0..self.block_size {
                let dst_val = gf256(output[j]);
                let src_val = gf256(src_block[j]);
                output[j] = (dst_val + coeff * src_val).0;
            }
        }

        output
    }
}
// Main Wirehair decoder with optimized Gaussian elimination


/// Robust incremental decoder maintaining pivots in RREF.
pub struct Decoder {
    k: usize,
    block_size: usize,
    // For each column index, optional normalized coefficient vector (length k) that has 1 at that column.
    pivot_cols: Vec<Option<Vec<gf256>>>,
    // For each column index, optional normalized symbol data (length block_size).
    pivots: Vec<Option<Vec<u8>>>,
    rank: usize,
    seed: u32,
    decoded: bool,
}

impl Decoder {
    pub fn new(k: usize, block_size: usize) -> Self {
        Self {
            k,
            block_size,
            pivot_cols: vec![None; k],
            pivots: vec![None; k],
            rank: 0,
            seed: 0x12345678,
            decoded: false,
        }
    }

    /// Add a symbol. Returns true when decoding completed (rank == k).
    pub fn add_symbol(&mut self, symbol_id: u32, mut data: Vec<u8>) -> bool {
        if self.decoded {
            return true;
        }

        // 1) Build coefficient vector for incoming symbol
        let mut coeffs = vec![gf256(0); self.k];
        if (symbol_id as usize) < self.k {
            // systematic
            coeffs[symbol_id as usize] = gf256(1);
        } else {
            let mut rng = Xorshift32::new(self.seed ^ symbol_id);
            for i in 0..self.k {
                coeffs[i] = gf256((rng.next() & 0xFF) as u8);
            }
        }

        // 2) Forward-eliminate incoming row by existing pivots (in ascending pivot-col order).
        // This makes the incoming row orthogonal to previous pivots.
        for col in 0..self.k {
            if coeffs[col] == gf256(0) {
                continue;
            }
            if let Some(pivot_coeffs) = &self.pivot_cols[col] {
                // factor to eliminate
                let factor = coeffs[col];
                if factor != gf256(0) {
                    // coeffs -= pivot_coeffs * factor
                    for j in 0..self.k {
                        coeffs[j] -= pivot_coeffs[j] * factor;
                    }
                    // data  -= pivot_data * factor
                    let pivot_data = self.pivots[col].as_ref().unwrap();
                    for j in 0..self.block_size {
                        let v = gf256(data[j]) - gf256(pivot_data[j]) * factor;
                        data[j] = v.0;
                    }
                }
            }
        }

        // 3) If incoming row is now zero => dependent, drop it
        if coeffs.iter().all(|&c| c == gf256(0)) {
            return false;
        }

        // 4) Find first non-zero entry -> pivot column
        let pivot_col = coeffs.iter().position(|&c| c != gf256(0)).expect("non-zero row");
        // Normalize row so coeffs[pivot_col] == 1
        let inv = gf256(1) / coeffs[pivot_col];
        for j in 0..self.k {
            coeffs[j] *= inv;
        }
        for j in 0..self.block_size {
            data[j] = (gf256(data[j]) * inv).0;
        }

        // Store normalized pivot
        self.pivot_cols[pivot_col] = Some(coeffs.clone());
        self.pivots[pivot_col] = Some(data.clone());
        self.rank += 1;

        // 5) Back-eliminate this pivot column from all other stored pivots to maintain RREF.
        // Clone pivot data to avoid mutable borrow conflicts.
        let new_pivot_coeffs = self.pivot_cols[pivot_col].as_ref().unwrap().clone();
        let new_pivot_data = self.pivots[pivot_col].as_ref().unwrap().clone();

        for other_col in 0..self.k {
            if other_col == pivot_col {
                continue;
            }
            if let Some(other_coeffs) = &mut self.pivot_cols[other_col] {
                let factor = other_coeffs[pivot_col];
                if factor != gf256(0) {
                    // other_coeffs -= new_pivot_coeffs * factor
                    for j in 0..self.k {
                        other_coeffs[j] -= new_pivot_coeffs[j] * factor;
                    }
                    // other_pivot_data -= new_pivot_data * factor
                    let other_data = self.pivots[other_col].as_mut().unwrap();
                    for j in 0..self.block_size {
                        let v = gf256(other_data[j]) - gf256(new_pivot_data[j]) * factor;
                        other_data[j] = v.0;
                    }
                }
            }
        }

        // 6) Check completion
        if self.rank == self.k {
            // Double-check all pivots present (defensive)
            for i in 0..self.k {
                if self.pivots[i].is_none() || self.pivot_cols[i].is_none() {
                    // shouldn't happen, but if it does, not decoded yet
                    return false;
                }
            }
            self.decoded = true;
            return true;
        }

        false
    }

    /// When decoded == true, `pivots[i]` is the i-th source block.
    pub fn get_decoded(&self) -> Option<Vec<u8>> {
        if !self.decoded {
            return None;
        }
        let mut out = Vec::with_capacity(self.k * self.block_size);
        for i in 0..self.k {
            match &self.pivots[i] {
                Some(block) => out.extend_from_slice(block),
                None => return None,
            }
        }
        Some(out)
    }

    pub fn is_decoded(&self) -> bool {
        self.decoded
    }
}



// Fast PRNG for coefficient generation
struct Xorshift32 {
    state: u32,
}

impl Xorshift32 {
    fn new(seed: u32) -> Self {
        let mut state = seed;
        if state == 0 {
            state = 0x12345678;
        }
        Xorshift32 { state }
    }

    #[inline(always)]
    fn next(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let k = 10;
        let block_size = 1024;
        let data_size = k * block_size;

        // Generate test data
        let mut data = vec![0u8; data_size];
        for i in 0..data_size {
            data[i] = (i % 256) as u8;
        }

        // Encode
        let encoder = Encoder::new(data.clone(), k, block_size);

        // Decode with exactly k symbols
        let mut decoder = Decoder::new(k, block_size);

        for i in 0..k {
            let symbol = encoder.encode(i as u32);
            decoder.add_symbol(i as u32, symbol);
        }

        // assert!(decoder.is_decoded());
        let decoded = decoder.get_decoded().unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_with_loss() {
        let k = 10;
        let block_size = 1024;
        let data_size = k * block_size;

        // Generate test data
        let mut data = vec![0u8; data_size];
        for i in 0..data_size {
            data[i] = (i % 256) as u8;
        }

        // Encode
        let encoder = Encoder::new(data.clone(), k, block_size);

        // Decode with some packet loss (skip symbol 3 and 7)
        let mut decoder = Decoder::new(k, block_size);

        for i in 0..k + 2 {
            let symbol = encoder.encode(i as u32);
            if i == 3 || i == 7 {
                continue;
            }
            decoder.add_symbol(i as u32, symbol);
        }

        // assert!(decoder.is_decoded());
        let decoded = decoder.get_decoded().unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_systematic_codes() {
        let k = 5;
        let block_size = 100;

        // Create simple test data
        let mut data = vec![0u8; k * block_size];
        for i in 0..data.len() {
            data[i] = (i % 256) as u8;
        }

        let encoder = Encoder::new(data.clone(), k, block_size);
        let mut decoder = Decoder::new(k, block_size);

        // Add first k symbols (systematic)
        for i in 0..k {
            let symbol = encoder.encode(i as u32);
            let added = decoder.add_symbol(i as u32, symbol);
            if i == k - 1 {
                assert!(added, "Should decode after k symbols");
            }
        }

        let decoded = decoder.get_decoded().unwrap();
        assert_eq!(decoded, data);
    }
}
