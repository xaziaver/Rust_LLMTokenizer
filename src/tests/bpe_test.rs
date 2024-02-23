use std::collections::HashMap;

// instead of using the raw UTF-8 bytes we want to support a larger vocabulary size
// that we can tune as a hyperparameter while sticking with the same encoding
pub fn execute(test_string: &str) -> (Vec<u32>, HashMap<(u32,u32), u32>, HashMap<u32, (u32,u32)>) {
    let increase_vocab_size = 20;
    let (bpe_vec, encode_map, decode_map) = train_tokenizer(test_string.as_bytes().to_vec(), increase_vocab_size);
    (bpe_vec, encode_map, decode_map)
}

// the BPE algorithm: compresses `utf8_bytes` while adding to the vocabularly by `increase`
// https://en.wikipedia.org/wiki/Byte_pair_encoding
fn train_tokenizer(utf8_bytes: Vec<u8>, increase: u32) -> (Vec<u32>, HashMap<(u32,u32), u32>, HashMap<u32, (u32,u32)>) {
    let mut encode_map: HashMap<(u32, u32), u32> = HashMap::new();
    let mut decode_map: HashMap<u32, (u32, u32)> = HashMap::new();
    let mut compressed_bytes: Vec<u32> = utf8_bytes.into_iter().map(|val| val as u32).collect();
    let mut new_word: u32 = 256;    // start with 256 as the first 'new' word after the initial byte range

    for _ in 0..increase {
        if let Some(((byte1, byte2), _)) = find_most_frequent_pair(&compressed_bytes) {
            // update maps
            encode_map.insert((byte1, byte2), new_word);
            decode_map.insert(new_word, (byte1, byte2));
            println!("minting ({}, {}) into a new token {}", byte1, byte2, new_word);

            // compress vector by replacing all occurrences of pair with new_word
            let mut i = 0;
            while i < compressed_bytes.len() - 1 {
                if compressed_bytes[i] == byte1 as u32 && compressed_bytes[i + 1] == byte2 as u32 {
                    compressed_bytes[i] = new_word;
                    // find new way to update.. inefficient here
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
    println!("extended vocabulary from 256 words to {}", new_word); 
    (compressed_bytes, encode_map, decode_map)
}

fn find_most_frequent_pair(input_vec: &Vec<u32>) -> Option<((u32, u32), usize)> {
    
    let mut pair_counts = HashMap::new();
    // iterate over each pair and count occurrences
    for window in input_vec.windows(2) {
        if let [a, b] = *window {
            let pair = ((a, b), 1);
            *pair_counts.entry(pair.0).or_insert(0) += pair.1;
        }
    }
    // find the pair with the maximum count (at least 2)
    pair_counts.into_iter()
        .filter(|&(_, count)| count >= 2)
        .max_by_key(|&(_, count)| count)
} 