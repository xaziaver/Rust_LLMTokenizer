// TODO: look at available inference source code (https://github.com/openai/tiktoken) to try to reverse
// engineer training methods... tiktoken/tiktoken_ext/openai_public.py shows some details
// tiktokenizer.vercel.app to see results with different versions of tokenizers

use std::fs::File;
use std::io::Write;
use std::collections::HashMap;
use regex::Regex;

pub struct Vocabulary {
    pub vocab_hash: HashMap<u32, (u32, u32)>,    // for decoding
    pub vocab_vec: Vec<((u32, u32), u32)>,      // for encoding
}

impl Vocabulary {

    /// convert a sequence of u32 values to a UTF-8 encoded string
    /// w/ special handling for control characters and invalid sequences
    pub fn stringify_word(&self, bytes: &[u32]) -> String {
        let expanded_bytes = self.expand_bytes(bytes);
        let mut result_string = String::new();
        let mut last_was_invalid = false;
        
        // iterate over each byte in the expanded sequence
        for byte in expanded_bytes.iter() {

            //  check for control and escape characters
            if *byte < 0x20 || *byte == 0x7F {
                // to prevent duplicating escape sequences 
                if !last_was_invalid {
                    result_string.push_str(&format!("\\u{:04x}", byte));
                    last_was_invalid = true;
                }

            // valid utf-8 character then push
            } else if *byte < 0x7F {
                result_string.push(*byte as u8 as char);
                last_was_invalid = false;

            //  invalid utf-8 then replace once
            } else {
                if !last_was_invalid {
                    result_string.push('\u{FFFD}');
                    last_was_invalid = true;
                }
            }
        }

        result_string
    }

    /// helper for stringify_word()... expand bytes that have 
    /// been encoded into u32 values to get the original sequence
    fn expand_bytes(&self, bytes: &[u32]) -> Vec<u8> {
        let mut result: Vec<u8> = Vec::new();
        
        for &byte in bytes {
            // directly convert
            if byte <= 255 {
                result.push(byte as u8);
            
            // recursively unpack the "extended byte" into merged u8 values using the mapping
            } else {
                if let Some(&(byte1, byte2)) = self.vocab_hash.get(&byte) {
                    let expanded_bytes1 = self.expand_bytes(&[byte1]);
                    let expanded_bytes2 = self.expand_bytes(&[byte2]);
                    result.extend(expanded_bytes1);
                    result.extend(expanded_bytes2);
                }
            }
        }

        result
    }
    
}


// instead of using the raw UTF-8 bytes we want to support a larger vocabulary size
// that we can tune as a hyperparameter while sticking with the same encoding
pub fn execute(test_string: &str, verbose: bool) -> Vocabulary {
    let target_vocab_size = 512;
    let new_vocabulary = train_tokenizer(test_string, target_vocab_size, verbose);
    
    new_vocabulary
}


/// uses the BPE algorithm to merge the most common pairs of bytes across chunks of the input text
/// the number of merges depends on the desired 'target' words in the returned Vocabulary object
fn train_tokenizer(text: &str, target: u32, verbose: bool) -> Vocabulary {
    let mut file = File::create("data/output/train_output.txt").unwrap();
    let mut file2 = File::create("data/output/chunk_output.txt").unwrap();
    let mut vocab = Vocabulary {
        vocab_hash: HashMap::<u32, (u32, u32)>::new(),
        vocab_vec: Vec::<((u32, u32), u32)>::new(),
    };
    // start with 256 as the first new 'word' after the initial byte range
    let mut new_word: u32 = 256;
    let total_merges = target-new_word;

    // split, convert to bytes, then extend the bytes to hold new words
    let split_text: Vec<String> = split(text);
    
    // for checking chunking
    for chunk in &split_text {
        writeln!(file2, "{}", chunk).unwrap();
    }
    
    let mut split_bytes_ext: Vec<Vec<u32>> = 
        split_text
            .iter()
            .map(|s| s.as_bytes().iter().map(|&b| b as u32).collect())
            .collect();

    let mut pairs: HashMap<(u32, u32), u32> = HashMap::new();
    let mut merges = total_merges;
    // for each loop.. merge 1 byte sequence and get 1 new word
    while merges > 0 {
        pairs.clear();
        // to determine what to merge.. find the most common pair of 
        // consecutive bytes when each text chunk is considered separately
        for chunk in &split_bytes_ext {
            let chunk_pairs = pair_counts(chunk);
            for (pair, count) in chunk_pairs {
                *pairs.entry(pair).or_insert(0) += count;
            }
        }

        if let Some(((byte1, byte2), count)) = pairs.iter()
            // find the pair with the maximum count (at least 2)
            // ties are broken by lexiographical order
                .filter(|&(_, count)| *count >= 2)
                .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
                .map(|(pair, &count)| (*pair, count)) {

            // update Vocabulary
            vocab.vocab_vec.push(((byte1, byte2), new_word));
            vocab.vocab_hash.insert(new_word, (byte1, byte2));
            
            // print most common pair found across all chunks and the new word
            let string_view = vocab.stringify_word(&[byte1, byte2]);
            if verbose == true {
                println!("merge {}/{}: ({}, {}) -> {} (b'{}') had {} occurrences",
                    total_merges-merges+1, total_merges, byte1, byte2, new_word, string_view, count);
             }
            writeln!(file, "[{}][{}] -> [{}] {}", 
                vocab.stringify_word(&[byte1]), 
                vocab.stringify_word(&[byte2]), 
                &string_view, 
                &new_word)
            .unwrap();

            // replace all occurrences of pair in each chunk with new_word
            for chunk in &mut split_bytes_ext {
                let mut i = 0;
                while i < chunk.len() - 1 {
                    // find new way to update.. inefficient here
                    if chunk[i] == byte1 && chunk[i + 1] == byte2 {
                        chunk[i] = new_word;
                        chunk.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
           new_word += 1; // prepare the new word for the next iteration
        } else {
           break;  // break if no more pairs are found
        }
        merges -= 1;
    }
    println!("extended vocabulary by {} from 256 words to {}", new_word-256, new_word); 
    
    vocab
}


/// returns counts for consecutive element pairs' occurrences.
pub fn pair_counts(input_vec: &Vec<u32>) -> HashMap<(u32, u32), u32> {
    
    let mut pair_counts = HashMap::new();
    for window in input_vec.windows(2) {
        if let [a, b] = *window {
            let pair = ((a, b), 1);
            *pair_counts.entry(pair.0).or_insert(0) += pair.1;
        }
    }

    pair_counts
}


/// splits text into chunks and returns those in a vector
pub fn split(text: &str)  -> Vec<String> {
    // adaptation of pattern used for GPT-4 tokenizer
    let combined_pattern = format!(
        r"{}|{}|{}|{}|{}|{}",
        r"(?i:'(?:[sdmt]|ll|ve|re))",               // contractions (new syntax, but should match?)
        r"[^\r\n\p{L}\p{N}]\p{L}+|^\p{L}+",         // words (does not include ?+, tried to recreate)
        r"\p{N}{1,3}",                              // numbers (same syntax)
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",              // special characters  (+ instead of ++, but should match?)
        r"\s*[\r\n]",                              // newlines (same syntax)
        r"\s+"                                     // spaces (matches after adjust_whitespace())
    );
    let regex = Regex::new(&combined_pattern).unwrap();

    let mut results = Vec::new();
    let mut last_end = 0;
    for mat in regex.find_iter(text) {
        if last_end != mat.start() {
            // push text between matches
            results.push(text[last_end..mat.start()].to_string());
        }
        // push match
        results.push(mat.as_str().to_string());
        last_end = mat.end();
    }
    // push any remaining text after last match
    if last_end < text.len() {
        results.push(text[last_end..].to_string());
    }

    adjust_whitespace(results)
}


/// helper for split(): handles the negative lookahead for white-space pattern: "\s+(?!\S)"
/// in the context of our pattern in split() since Rust Regex impl does not support
fn adjust_whitespace(tokens: Vec<String>) -> Vec<String> {
    let mut adjusted_tokens = Vec::new();
    let mut previous_was_whitespace = false;
    let is_whitespace = |s: &str| s.chars().all(|c| c == ' ' || c == '\t');
    
    for (i, token) in tokens.iter().enumerate() {

        // check if all whitespace
        if is_whitespace(token) {
            // check if next token starts w/ non-whitespace (negative lookahead)
            if i + 1 < tokens.len() && tokens[i + 1].starts_with(|c: char| !c.is_whitespace()) {
                // push all but the last whitespace, which
                // should be at the start of the next token
                if token.len() > 1 {
                    previous_was_whitespace = true;
                    adjusted_tokens.push(token[0..token.len() - 1].to_string());
                // no changes
                } else {
                    adjusted_tokens.push(token.clone());
                }
            // no changes
            } else {
                adjusted_tokens.push(token.clone());
            }
        
        // not all whitespace
        } else {
            // include the trailing whitespace from the
            // previous token at the beginning of this token
            if previous_was_whitespace {
                let new_token = format!(" {}", token);
                adjusted_tokens.push(new_token);
                previous_was_whitespace = false;
            // no changes
            } else {
                adjusted_tokens.push(token.clone());
            }
        }
    }

    adjusted_tokens
}


// TODO: incorporate special tokens after researching & determining what's needed
//       (requires changes to transformer's code to account for new special tokens)

// TODO: explore new types of tokens like those in 
//       "Learning to Compress Prompts with Gist Tokens" by Jesse Mu, Xiang Lisa Li, Noah Goodman

/*
SPECIAL TOKENS
--------------
In addition to tokens coming from the bpe merges, special tokens can be inserted into
the output in order to delimit or provide some structure for the LLM.

For example, the GPT-2 tokenizer had a vocab length of 50257..
    50257 = 256 raw byte tokens + 50,000 merges +1 special token
the special token here is '<|endoftext|>' which OpenAI added as a
way to separate documents when training the LLM.

The training set consisted of many documents which were each converted 
to token streams with values between 0-50256 & with 50257 added at the end...
the idea was for the LLM to eventually learn this means the documents ends
and the data proceeding is unrelated.

There are more special tokens relating to not only the training, but prompting as well
(check previously cited source code, e.g.):
 - FIM_PREFIX = "<|fim_prefix|>"
 - FIM_MIDDLE = "<|fim_middle|>"
 - FIM_SUFFIX = "<|fim_suffix|>"
"Fill In Middle" is described in the paper "Efficient Training of Language Models to Fill in the Middle"
28 Jul 2022 arxiv.org
*/