Wirehair implementation in Rust, currently 2x as slow as the C++ version.

Usage:

```use wirehair::*;
fn main()   {
    let message = b"This crate is a Rust port of the Google Congestion Control (GCC) implementation from the WebRTC project. No more painful linking against libwebrtc.

Google Congestion Control is the congestion control algorithm used in WebRTC. It is a hybrid algorithm that combines the strengths of delay-based and loss-based congestion control. The end goal is minimum latency and to avoid buffer-bloat as much as possible.

All licensing from the original source code is preserved under a BSD-style license. Thanks to the WebRTC project authors for their hard work on this implementation. All I did was port it to Rust";

    // Choose a block size (e.g. 32 bytes per block)
    let block_size_bytes: u32 = 32;
    let now = std::time::Instant::now();
    // 2. Create encoder
    let encoder = Encoder::new(message.to_vec(), 22, block_size_bytes as usize);
    let mut decoder = Decoder::new(22, block_size_bytes as usize);
    for block_id in 0..25 {
        let block_buf = encoder.encode(block_id);
        // block_buf.truncate(block_out_bytes as usize);
        if block_id %10 == 0 {
            continue;
        }
        decoder.add_symbol(block_id as u32, block_buf);
        if decoder.is_decoded() {
            println!("Decoder is ready after adding block {}", block_id);
            break;
        }
    }
    let recovered = decoder.get_decoded().unwrap();
    println!("Decoding took: {:?}", now.elapsed());
    println!("Recovered message: {}", String::from_utf8_lossy(&recovered));
}
```