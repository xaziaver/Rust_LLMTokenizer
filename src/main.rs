/*
Q: What do we want to do?
A: take strings and feed them into language models

For that we need to
1. "tokenize" the strings into a set of integers in some fixed vocabulary
2. those integers will be used to make a lookup into a table of vectors
3. the result vectors will be fed into the transformer as an input

-----------LLM----------
-----token sequence-----
     ^             |
     |  Tokenizer  |
     |             v
--------raw text--------

The tokenizer has its own training set used in a preprocessing stage
to train the vocabulary before being used with the LLM. This also provides
a mapping which can be used to quickly translate between 

The selection of training data for the tokenizer can vary depending on your goals,
the important thing to understand is the more of a certain type of text is used,
the better the LLM is at understanding it... This is because there will be more merges 
during compression of that type and so the token sequences will be shorter which
works better for the LLM's finite context length

*/
use std::fs;
use std::collections::HashMap;
use std::collections::HashSet;

pub mod training;


fn main() -> Result<(), std::io::Error> {
    // fetch data and run training to get maps
    let data_file_path = "data/unicode_blog.txt";
    let data = fs::read_to_string(data_file_path)?;
    let training_set: &str = &data;
    
    println!("");
    println!("START TRAINING");
    println!("##############################");
    let tokenizer = crate::training::execute(training_set);
    println!("##############################");
    println!("END TRAINING");
    println!("");

    
    let example_file_path = "data/example.txt";
    let example = fs::read_to_string(example_file_path)?;
    let text_example: &str = &example;
    println!("");
    println!("START ENCODING");
    println!("##############################");
    let text_encoded = encode(tokenizer.vocab_vec, text_example.as_bytes().to_vec());
    let text_decoded = decode(tokenizer.vocab_hash, text_encoded);
    println!("##############################");
    println!("END ENCODING");
    println!("");

    println!("text after encoding and decoding: \n{}", text_decoded);
    println!("");
    assert_eq!(text_example.as_bytes().to_vec(), text_decoded.as_bytes().to_vec());
    println!("if you can see this then they were byte-wise equal!");
    println!("");
 
    Ok(())
}

fn encode(encode_map: Vec<((u32, u32), u32)>, text_vector: Vec<u8>) -> Vec<u32> {
    let text_clone = text_vector.clone();
    // translate into Vec<u32> to hold the extended bytes
    let mut text: Vec<u32> = text_vector.into_iter().map(|val| val as u32).collect();
    
    for (pair, ext_byte) in &encode_map {
        let pairs = crate::training::pair_counts(&text);
        let pairs_set: HashSet<&(u32, u32)> = pairs.keys().collect();

        if pairs_set.contains(&pair) {
            let (byte1, byte2) = pair;
            let mut i = 0;
            while i < text.len() - 1 {
                if text[i] == *byte1 && text[i + 1] == *byte2 {
                    println!("replacing `{:?}{:?}` with extended byte {}", char::from_u32(*byte1).unwrap_or('?'), char::from_u32(*byte2).unwrap_or('?'), ext_byte);
                    text[i] = *ext_byte;
                    text.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
    }

    println!("encoding compression ratio: {}", text_clone.len() as f32 / text.len() as f32);
    text
}

// TODO: take care of case when LLM provides invalid byte sequences (replace with invalid char)
fn decode(decode_map: HashMap<u32, (u32, u32)>, tokens_vector: Vec<u32>) -> String {
    let mut tokens = tokens_vector;
    let mut i = 0;
    while i < tokens.len() {
        if let Some(&(byte1, byte2)) = decode_map.get(&tokens[i]) {
            //println!("Replacing {} with ({}, {})", &tokens[i], byte1, byte2);
            tokens[i] = byte1;              // replace current token with its first component
            tokens.insert(i+1, byte2);     // insert second component next to first
            continue;                      // allows inserts to be checked
        }
        i += 1;  // move to the next token
    }
    // Convert tokens back to bytes and then to a String
    // Assuming all tokens are now in the original byte range 0-255
    let decoded_bytes: Vec<u8> = tokens.into_iter().map(|token| token as u8).collect();
    String::from_utf8(decoded_bytes).unwrap_or_else(|_| String::from("Decoding Error"))
}