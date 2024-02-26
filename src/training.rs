use std::collections::HashMap;

pub struct Vocabulary {
    pub vocab_hash: HashMap<u32, (u32, u32)>,    // for decoding
    pub vocab_vec: Vec<((u32, u32), u32)>,      // for encoding
}

// instead of using the raw UTF-8 bytes we want to support a larger vocabulary size
// that we can tune as a hyperparameter while sticking with the same encoding
pub fn execute(test_string: &str) -> Vocabulary {
    let increase_vocab_size = 200;
    let new_vocabulary = train_tokenizer(test_string.as_bytes().to_vec(), increase_vocab_size);
    new_vocabulary
}

// the BPE algorithm: compresses `utf8_bytes` while adding to the vocabularly by `increase`
// https://en.wikipedia.org/wiki/Byte_pair_encoding

// TODO: improve new token creation w/ GPT2 methods: prevent seperate merges for (e.g. dog, dog., dog!, dog?, ..)
fn train_tokenizer(utf8_bytes: Vec<u8>, increase: u32) -> Vocabulary {
    let mut vocab = Vocabulary {
        vocab_hash: HashMap::<u32, (u32, u32)>::new(),
        vocab_vec: Vec::<((u32, u32), u32)>::new(),
    };
    let mut extended_bytes: Vec<u32> = utf8_bytes.into_iter().map(|val| val as u32).collect();
    let mut new_word: u32 = 256;    // start with 256 as the first 'new' word after the initial byte range

    for _ in 0..increase {
        if let Some(((byte1, byte2), _)) = 
            // find the pair with the maximum count (at least 2)
            pair_counts(&extended_bytes).into_iter()
                .filter(|&(_, count)| count >= 2)
                .max_by_key(|&(_, count)| count) {

           // update Vocabulary
           vocab.vocab_vec.push(((byte1, byte2), new_word));
           vocab.vocab_hash.insert(new_word, (byte1, byte2));
           println!("minting ({}, {}) into a new token {}", byte1, byte2, new_word);

           // compress vector by replacing all occurrences of pair with new_word
           let mut i = 0;
           while i < extended_bytes.len() - 1 {
               if extended_bytes[i] == byte1 as u32 && extended_bytes[i + 1] == byte2 as u32 {
                   extended_bytes[i] = new_word;
                   // find new way to update.. inefficient here
                   extended_bytes.remove(i + 1);
               } else {
                   i += 1;
               }
           }

           new_word += 1; // prepare the new word for the next iteration
        } else {
           break;  // break if no more pairs are found
        }
    }
    println!("extended vocabulary from 256 words to {}", new_word); 
    vocab
}

pub fn pair_counts(input_vec: &Vec<u32>) -> HashMap<(u32, u32), u32> {
    
    let mut pair_counts = HashMap::new();
    // iterate over each pair and count occurrences
    for window in input_vec.windows(2) {
        if let [a, b] = *window {
            let pair = ((a, b), 1);
            *pair_counts.entry(pair.0).or_insert(0) += pair.1;
        }
    }
    pair_counts
}