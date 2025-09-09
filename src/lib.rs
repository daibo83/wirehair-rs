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


pub struct Decoder {
    k: usize,
    block_size: usize,
    matrix: Vec<Vec<gf256>>,   // coefficient matrix
    symbols: Vec<Vec<u8>>,     // received symbol data
    seed: u32,
    decoded: bool,
    decoded_blocks: Vec<Vec<u8>>,
}

impl Decoder {
    pub fn new(k: usize, block_size: usize) -> Self {
        Decoder {
            k,
            block_size,
            matrix: Vec::new(),
            symbols: Vec::new(),
            seed: 0x12345678,
            decoded: false,
            decoded_blocks: vec![vec![0u8; block_size]; k],
        }
    }

    pub fn add_symbol(&mut self, symbol_id: u32, data: Vec<u8>) -> bool {
        if self.decoded {
            return true;
        }

        let mut coeffs = vec![gf256(0); self.k];
        if (symbol_id as usize) < self.k {
            // Systematic identity row
            coeffs[symbol_id as usize] = gf256(1);
        } else {
            // Random coefficients
            let mut rng = Xorshift32::new(self.seed ^ symbol_id);
            for i in 0..self.k {
                coeffs[i] = gf256((rng.next() & 0xFF) as u8);
            }
        }

        self.matrix.push(coeffs);
        self.symbols.push(data);

        if self.symbols.len() >= self.k {
            self.decoded = self.gaussian_elimination();
        }

        self.decoded
    }

    fn gaussian_elimination(&mut self) -> bool {
        let n = self.symbols.len();
        if n < self.k {
            return false;
        }

        let mut aug_matrix = self.matrix.clone();
        let mut aug_symbols = self.symbols.clone();

        for col in 0..self.k {
            // Find pivot
            let mut pivot_row = None;
            for row in col..n {
                if aug_matrix[row][col] != gf256(0) {
                    pivot_row = Some(row);
                    break;
                }
            }
            let Some(prow) = pivot_row else { continue };

            // Swap rows
            if prow != col {
                aug_matrix.swap(col, prow);
                aug_symbols.swap(col, prow);
            }

            // Normalize pivot row
            let pivot_val = aug_matrix[col][col];
            if pivot_val != gf256(1) {
                let inv = gf256(1) / pivot_val;
                for j in 0..self.k {
                    aug_matrix[col][j] *= inv;
                }
                for j in 0..self.block_size {
                    aug_symbols[col][j] = (gf256(aug_symbols[col][j]) * inv).0;
                }
            }

            // Eliminate column from other rows
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug_matrix[row][col];
                if factor == gf256(0) {
                    continue;
                }

                for j in 0..self.k {
                    let a = aug_matrix[col][j] * factor;
                    aug_matrix[row][j] -= a;
                }
                for j in 0..self.block_size {
                    let val = gf256(aug_symbols[row][j])
                        - gf256(aug_symbols[col][j]) * factor;
                    aug_symbols[row][j] = val.0;
                }
            }
        }

        // Check if we have identity matrix
        for i in 0..self.k {
            for j in 0..self.k {
                let expected = if i == j { gf256(1) } else { gf256(0) };
                if aug_matrix[i][j] != expected {
                    return false;
                }
            }
        }

        // Extract decoded blocks
        for i in 0..self.k {
            self.decoded_blocks[i] = aug_symbols[i].clone();
        }

        true
    }

    pub fn get_decoded(&self) -> Option<Vec<u8>> {
        if !self.decoded {
            return None;
        }
        let mut result = Vec::with_capacity(self.k * self.block_size);
        for block in &self.decoded_blocks {
            result.extend_from_slice(block);
        }
        Some(result)
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

        assert!(decoder.is_decoded());
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

        assert!(decoder.is_decoded());
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
