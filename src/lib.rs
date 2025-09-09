// Wirehair Forward Error Correction Implementation
// Optimized for speed with SIMD, cache-friendly layouts, and minimal allocations

use std::arch::x86_64::*;

// Optimized Galois Field 256 operations
mod gf256 {
    use super::*;

    static mut LOG_TABLE: [u8; 256] = [0; 256];
    static mut EXP_TABLE: [u8; 512] = [0; 512];
    static mut MUL_TABLE: [[u8; 256]; 256] = [[0; 256]; 256];
    static INIT: std::sync::Once = std::sync::Once::new();

    pub fn init() {
        INIT.call_once(|| unsafe {
            // Generate exp/log tables for GF(256) with polynomial 0x11d
            let mut x = 1u8;
            for i in 0..255 {
                EXP_TABLE[i] = x;
                EXP_TABLE[i + 255] = x;
                LOG_TABLE[x as usize] = i as u8;

                // Multiply by generator (2)
                x = (x << 1) ^ if x & 0x80 != 0 { 0x1d } else { 0 };
            }
            LOG_TABLE[0] = 0;

            // Precompute multiplication table
            for a in 0..256 {
                for b in 0..256 {
                    if a == 0 || b == 0 {
                        MUL_TABLE[a][b] = 0;
                    } else {
                        let log_sum = (LOG_TABLE[a] as u16 + LOG_TABLE[b] as u16) % 255;
                        MUL_TABLE[a][b] = EXP_TABLE[log_sum as usize];
                    }
                }
            }
        });
    }

    #[inline(always)]
    pub fn mul(a: u8, b: u8) -> u8 {
        unsafe { MUL_TABLE[a as usize][b as usize] }
    }

    #[inline(always)]
    pub fn inv(a: u8) -> u8 {
        unsafe {
            if a == 0 {
                return 0;
            }
            EXP_TABLE[255 - LOG_TABLE[a as usize] as usize]
        }
    }

    // SIMD multiplication and add using AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn mul_add_avx2(dst: *mut u8, src: *const u8, val: u8, len: usize) {
        if val == 0 {
            return;
        }

        // For small lengths or val == 1, use scalar
        if len < 32 || val == 1 {
            for i in 0..len {
                unsafe { *dst.add(i) ^= if val == 1 {
                    *src.add(i)
                } else {
                    mul(*src.add(i), val)
                } };
            }
            return;
        }

        let chunks = len / 32;
        let remainder = len % 32;

        // Build lookup tables for SIMD multiplication
        // Build 16-entry nibble LUTs and duplicate to 32 bytes (one per 128-bit lane)
        let mut lo_table32 = [0u8; 32];
        let mut hi_table32 = [0u8; 32];
        for i in 0..16 {
            let lo = mul(i as u8, val);
            let hi = mul(((i as u8) << 4) as u8, val);
            lo_table32[i] = lo;
            lo_table32[i + 16] = lo; // duplicate for upper lane
            hi_table32[i] = hi;
            hi_table32[i + 16] = hi; // duplicate for upper lane
        }

        let lo_lut = unsafe { _mm256_loadu_si256(lo_table32.as_ptr() as *const __m256i) };
        let hi_lut = unsafe { _mm256_loadu_si256(hi_table32.as_ptr() as *const __m256i) };
        let mask = _mm256_set1_epi8(0x0f);

        // Process 32 bytes at a time
        for i in 0..chunks {
            let offset = i * 32;
            let src_vec = unsafe { _mm256_loadu_si256(src.add(offset) as *const __m256i) };
            let dst_vec = unsafe { _mm256_loadu_si256(dst.add(offset) as *const __m256i) };

            // Split into nibbles
            let lo = _mm256_and_si256(src_vec, mask);
            let hi = _mm256_srli_epi16(src_vec, 4);
            let hi = _mm256_and_si256(hi, mask);

            // Table lookup multiplication
            let prod_lo = _mm256_shuffle_epi8(lo_lut, lo);
            let prod_hi = _mm256_shuffle_epi8(hi_lut, hi);
            let prod = _mm256_xor_si256(prod_lo, prod_hi);

            // XOR with destination
            let result = _mm256_xor_si256(dst_vec, prod);
            unsafe { _mm256_storeu_si256(dst.add(offset) as *mut __m256i, result) };
        }

        // Handle remainder
        for i in 0..remainder {
            let idx = chunks * 32 + i;
            unsafe { *dst.add(idx) ^= mul(*src.add(idx), val) };
        }
    }

    // Scalar multiplication and add
    pub fn mul_add_scalar(dst: &mut [u8], src: &[u8], val: u8) {
        if val == 0 {
            return;
        }

        let len = dst.len().min(src.len());
        if val == 1 {
            for i in 0..len {
                dst[i] ^= src[i];
            }
        } else {
            for i in 0..len {
                dst[i] ^= mul(src[i], val);
            }
        }
    }
}

// Main Wirehair encoder
pub struct Encoder {
    k: usize,          // Number of source blocks
    block_size: usize, // Size of each block
    data: Vec<u8>,     // Source data
    seed: u32,         // Random seed for encoding
}

impl Encoder {
    pub fn new(data: Vec<u8>, k: usize, block_size: usize) -> Self {
        gf256::init();

        // Ensure data is properly padded
        let mut padded_data = data;
        let expected_size = k * block_size;
        padded_data.resize(expected_size, 0);

        Encoder {
            k,
            block_size,
            data: padded_data,
            seed: 0x12345678, // Fixed seed for reproducibility
        }
    }

    // Generate encoding symbols using optimized random linear combinations
    pub fn encode(&self, symbol_id: u32) -> Vec<u8> {
        // Systematic first:
        if (symbol_id as usize) < self.k {
            let i = symbol_id as usize;
            return self.data[i * self.block_size..(i + 1) * self.block_size].to_vec();
        }

        let mut output = vec![0u8; self.block_size];
        let mut rng = Xorshift32::new(self.seed ^ symbol_id);

        for i in 0..self.k {
            let coeff = (rng.next() & 0xFF) as u8;
            if coeff == 0 {
                continue;
            }

            let src_block = &self.data[i * self.block_size..(i + 1) * self.block_size];

            unsafe {
                if is_x86_feature_detected!("avx2") {
                    gf256::mul_add_avx2(
                        output.as_mut_ptr(),
                        src_block.as_ptr(),
                        coeff,
                        self.block_size,
                    );
                } else {
                    gf256::mul_add_scalar(&mut output, src_block, coeff);
                }
            }
        }

        output
    }

    // Get coefficient for a specific source block in a symbol
    fn get_coefficient(&self, symbol_id: u32, block_idx: usize) -> u8 {
        let mut rng = Xorshift32::new(self.seed ^ symbol_id);
        for idx in 0..=block_idx {
            if block_idx == idx {
                return (rng.next() & 0xFF) as u8;
            }
            rng.next();
        }
        0
    }
}

// Main Wirehair decoder with optimized Gaussian elimination
pub struct Decoder {
    k: usize,
    block_size: usize,
    matrix: Vec<Vec<u8>>,  // Coefficient matrix
    symbols: Vec<Vec<u8>>, // Received symbol data
    symbol_ids: Vec<u32>,  // IDs of received symbols
    seed: u32,
    decoded: bool,
    decoded_blocks: Vec<Vec<u8>>, // Decoded source blocks
}

impl Decoder {
    pub fn new(k: usize, block_size: usize) -> Self {
        gf256::init();

        Decoder {
            k,
            block_size,
            matrix: Vec::new(),
            symbols: Vec::new(),
            symbol_ids: Vec::new(),
            seed: 0x12345678,
            decoded: false,
            decoded_blocks: vec![vec![0u8; block_size]; k],
        }
    }

    pub fn add_symbol(&mut self, symbol_id: u32, data: Vec<u8>) -> bool {
        if self.decoded {
            return true;
        }

        // Coefficients must match the encoder exactly:
        let mut coeffs = vec![0u8; self.k];
        if (symbol_id as usize) < self.k {
            // Systematic row e_i
            coeffs[symbol_id as usize] = 1;
        } else {
            let mut rng = Xorshift32::new(self.seed ^ symbol_id);
            for i in 0..self.k {
                coeffs[i] = (rng.next() & 0xFF) as u8;
            }
        }

        self.matrix.push(coeffs);
        self.symbols.push(data);
        self.symbol_ids.push(symbol_id);

        if self.symbols.len() >= self.k {
            self.decoded = self.gaussian_elimination();
        }

        self.decoded
    }

    // Perform Gaussian elimination to solve the system
    fn gaussian_elimination(&mut self) -> bool {
        let n = self.symbols.len();
        if n < self.k {
            return false;
        }

        // Create augmented matrix (coefficients | symbols)
        let mut aug_matrix = self.matrix.clone();
        let mut aug_symbols = self.symbols.clone();

        // Forward elimination to get row echelon form
        for col in 0..self.k {
            // Find pivot row with non-zero element in current column
            let mut pivot_row = None;
            for row in col..n {
                if aug_matrix[row][col] != 0 {
                    pivot_row = Some(row);
                    break;
                }
            }

            let Some(prow) = pivot_row else {
                // No pivot found, system is not solvable
                return false;
            };

            // Swap pivot row to current position
            if prow != col {
                aug_matrix.swap(col, prow);
                aug_symbols.swap(col, prow);
            }

            // Scale pivot row to have leading 1
            let pivot_val = aug_matrix[col][col];
            if pivot_val != 1 {
                let inv = gf256::inv(pivot_val);
                for j in col..self.k {
                    // Start from col instead of 0
                    aug_matrix[col][j] = gf256::mul(aug_matrix[col][j], inv);
                }
                for j in 0..self.block_size {
                    aug_symbols[col][j] = gf256::mul(aug_symbols[col][j], inv);
                }
            }

            // Eliminate column in all other rows
            for row in 0..n {
                if row == col {
                    continue;
                }

                let factor = aug_matrix[row][col];
                if factor == 0 {
                    continue;
                }

                // Subtract factor * pivot_row from current row
                for j in col..self.k {
                    // Start from col for efficiency
                    aug_matrix[row][j] ^= gf256::mul(aug_matrix[col][j], factor);
                }

                // Update symbol data
                for j in 0..self.block_size {
                    aug_symbols[row][j] ^= gf256::mul(aug_symbols[col][j], factor);
                }
            }
        }

        // Check if we have identity matrix in first k rows
        for i in 0..self.k {
            for j in 0..self.k {
                let expected = if i == j { 1 } else { 0 };
                if aug_matrix[i][j] != expected {
                    return false;
                }
            }
        }

        // If we have identity matrix, the first k symbols are the decoded blocks
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
    fn test_gf256_operations() {
        gf256::init();

        // Test multiplication properties
        assert_eq!(gf256::mul(0, 5), 0);
        assert_eq!(gf256::mul(5, 0), 0);
        assert_eq!(gf256::mul(1, 5), 5);
        assert_eq!(gf256::mul(5, 1), 5);

        // Test inverse
        for i in 1..256 {
            let inv = gf256::inv(i as u8);
            assert_eq!(gf256::mul(i as u8, inv), 1);
        }
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
