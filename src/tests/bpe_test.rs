use std::collections::HashMap;

// instead of using the raw UTF-8 bytes we want to support a larger vocabulary size
// that we can tune as a hyperparameter while sticking with the same encoding
pub fn execute(test_string: &str) -> Vec<u32> {
    let increase_vocab_size = 15;
    let bpe_vec: Vec<u32> = byte_pair_encoding(test_string.as_bytes().to_vec(), increase_vocab_size);
    bpe_vec
}

// the BPE algorithm allows us to compress our sequence of bytes
// https://en.wikipedia.org/wiki/Byte_pair_encoding
fn byte_pair_encoding(utf8_bytes: Vec<u8>, increase: u32) -> Vec<u32> {
    let mut compression_map: HashMap<(u8, u8), u32> = HashMap::new();
    let mut decompression_map: HashMap<u32, (u8, u8)> = HashMap::new();
    let mut compressed_bytes: Vec<u32> = utf8_bytes.into_iter().map(|val| val as u32).collect();
    let mut new_word: u32 = 256;    // start with 256 as the first 'new' word after the initial byte range

    for _ in 0..increase {
        if let Some(((byte1, byte2), _)) = find_most_frequent_pair(&compressed_bytes) {
            // update maps
            compression_map.insert((byte1, byte2), new_word);
            decompression_map.insert(new_word, (byte1, byte2));

            // compress vector by replacing all occurrences of pair with new_word
            let mut i = 0;
            while i < compressed_bytes.len() - 1 {
                if compressed_bytes[i] == byte1 as u32 && compressed_bytes[i + 1] == byte2 as u32 {
                    compressed_bytes[i] = new_word;
                    compressed_bytes.remove(i + 1);
                } else {
                    i += 1;
                }
            }

            new_word += 1; // prepare the new word for the next iteration
        } else {
            break;  // break if no more frequent pairs are found
        }
    }

    compressed_bytes
}

fn find_most_frequent_pair(input_vec: &Vec<u32>) -> Option<((u8, u8), usize)> {

    let mut pair_counts = HashMap::new();
    
    // iterate over each pair and count occurrences
    for window in input_vec.windows(2) {
        if let [a, b] = window {
            let pair = ((*a as u8, *b as u8), 1);
            *pair_counts.entry(pair.0).or_insert(0) += pair.1;
        }
    }

    // find the pair with the maximum count (at least 2)
    pair_counts.into_iter()
        .filter(|&(_, count)| count >= 2)
        .max_by_key(|&(_, count)| count)
} 