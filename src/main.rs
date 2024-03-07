// TODO: find some metric for improvements to training, encode, and decode functions

// TODO: explore sentencepiece library (both training and inference) used by LLama and Mistral
//       for an alternative approach (https://github.com/google/sentencepiece)


use std::fs;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use std::collections::HashMap;
use std::collections::HashSet;

pub mod training;


fn main() -> Result<(), std::io::Error> {
    // fetch data and run training to get maps
    let data_file_path = "data/train_text.txt";
    let data = fs::read_to_string(data_file_path)?;
    let training_set: &str = &data;

    // train tokenizer 
    println!("");
    println!("START TRAINING");
    println!("##############################");
    let start = Instant::now();
    let tokenizer = crate::training::execute(training_set, false);
    let duration = start.elapsed();
    println!("Training took {:.2} seconds", duration.as_secs_f64());
    println!("##############################");
    println!("END TRAINING");

    // encode example
    let example_file_path = "data/encode_text.txt";
    let example = fs::read_to_string(example_file_path)?;
    let text_example: &str = &example;
    println!("");
    println!("START ENCODING");
    println!("##############################");
    let text_encoded = encode(&tokenizer, text_example, false);
    
    // write output tokens to file
    let mut file = File::create("data/output/encode_output.txt")?;
    for token in &text_encoded {
        writeln!(file, "{}", tokenizer.stringify_word(&[*token]))?;
    }
    println!("##############################");
    println!("END ENCODING");

    // decode what was encoded for checking
    let text_decoded = decode(&tokenizer, text_encoded);
    
    // checks
    /*println!("");
    assert_eq!(text_example.as_bytes().to_vec(), text_decoded.as_bytes().to_vec());
    println!("if you can see this then they were byte-wise equal!");
    println!("");*/ 
    
    Ok(())
}


fn encode(vocab: &crate::training::Vocabulary, text: &str, verbose: bool) -> Vec<u32> {
    let start_len: usize = text.len();
    let end_len: usize;
    // split the text into chunks and translate to Vec<u32> to hold the extended bytes
    let split_text: Vec<String> = crate::training::split(text);
    let mut split_bytes_ext: Vec<Vec<u32>> = 
        split_text
            .iter()
            .map(|s| s.as_bytes().iter().map(|&b| b as u32).collect())
            .collect();
    let mut pairs_count: HashMap<(u32, u32), u32> = HashMap::new();
    let mut pairs_set: HashSet<(u32, u32)> = HashSet::new();
    let mut check_pairs = true;
    // check map in the order tokens were created
    for (pair, ext_byte) in &vocab.vocab_vec {
    
        // if output has been updated, get pairs across all chunks
        if check_pairs == true {
            for chunk in &split_bytes_ext {
                let chunk_pairs = crate::training::pair_counts(chunk);
                for (&key, &count) in chunk_pairs.iter() {
                    *pairs_count.entry(key).or_insert(0) += count;
                }
            }
            pairs_set = 
                pairs_count.iter()
                .filter(|&(_, &count)| count >= 2)
                .map(|(&key, _)| key)
                .collect();
        }
        
        // if the pair in the map is a pair in one of the chunks
        if pairs_set.contains(&pair) {
            // iterate through each chunk
            for chunk in &mut split_bytes_ext {
                let (byte1, byte2) = pair;
                let mut i = 0;
                // look for the pair in the chunk
                while i < chunk.len() - 1 {
                    if chunk[i] == *byte1 && chunk[i + 1] == *byte2 {
                        // print replacements
                        if verbose == true {
                            let string_byte1 = vocab.stringify_word(&[*byte1]);
                            let string_byte2 = vocab.stringify_word(&[*byte2]);
                            let string_view = format!("{}{}", &string_byte1, &string_byte2);
                            println!("replacing {:?}, {:?} with {:?}", 
                                string_byte1,
                                string_byte2,
                                string_view);
                        }
                        // replace the pair with the new word
                        chunk[i] = *ext_byte;
                        chunk.remove(i + 1);
                        check_pairs = true;
                    } else {
                        i += 1;
                    }
                }
            }
        } else {
            // no need to find pairs again
            // if no replacements were made
            check_pairs = false;
        }
    }    
    
    // de-chunk to prepare for output
    let mut encoded_text: Vec<u32> = vec!();
    for chunk in split_bytes_ext {
        let mut i = 0;
        while i < chunk.len() {
            encoded_text.push(chunk[i]);
            i += 1;
        }
    }
    end_len = encoded_text.len();
    println!("starting length: {},\nending length: {}", start_len, end_len);
    println!("encoding compression ratio: {}", start_len as f32 / end_len as f32);
    encoded_text
}


fn decode(vocab: &crate::training::Vocabulary, tokens_vector: Vec<u32>) -> String {
    let mut tokens = tokens_vector;
    let mut i = 0;
    while i < tokens.len() {
        if let Some(&(byte1, byte2)) = vocab.vocab_hash.get(&tokens[i]) {
            tokens[i] = byte1;                          // replace current token with its first component
            tokens.insert(i+1, byte2);   // insert second component next to first
            continue;                                   // allows inserts to be checked
        }
        i += 1;  // move to the next token
    }
    // tokens back to bytes and then string
    let decoded_bytes: Vec<u8> = tokens.into_iter().map(|token| token as u8).collect();
    String::from_utf8(decoded_bytes).unwrap_or_else(|_| String::from("Decoding Error"))
}