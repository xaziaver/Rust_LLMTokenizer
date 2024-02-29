use std::collections::HashMap;
use regex::Regex;

pub struct Vocabulary {
    pub vocab_hash: HashMap<u32, (u32, u32)>,    // for decoding
    pub vocab_vec: Vec<((u32, u32), u32)>,      // for encoding
    pub vocab_view: HashMap<u32, String>,       // to see string
}

impl Vocabulary {
    pub fn stringify_word(&self, byte: &u32) -> String { 
        if let Some(string) = self.vocab_view.get(byte) {
            string.clone() // Assuming you want to return the string as is.
        } else {
            // Convert the byte to a char and format it as a string, without debug formatting.
            char::from_u32(*byte).map(|c| c.to_string()).unwrap_or_default()
        }
    }
}

// instead of using the raw UTF-8 bytes we want to support a larger vocabulary size
// that we can tune as a hyperparameter while sticking with the same encoding
pub fn execute(test_string: &str) -> Vocabulary {
    let target_vocab_size = 500;
    let new_vocabulary = train_tokenizer(test_string, target_vocab_size);
    new_vocabulary
}


// TODO: look at available inference source code (https://github.com/openai/tiktoken) to try to reverse
// engineer training methods... tiktoken/tiktoken_ext/openai_public.py shows some details
// tiktokenizer.vercel.app


// the BPE algorithm: compresses `utf8_bytes` while increasing vocabulary to `target` 
// https://en.wikipedia.org/wiki/Byte_pair_encoding
fn train_tokenizer(text: &str, target: u32) -> Vocabulary {

    let mut vocab = Vocabulary {
        vocab_hash: HashMap::<u32, (u32, u32)>::new(),
        vocab_vec: Vec::<((u32, u32), u32)>::new(),
        vocab_view: HashMap::<u32, String>::new(),
    };
    // start with 256 as the first 'new' word after the initial byte range
    let mut new_word: u32 = 256;
    // split, convert to bytes, then extend the bytes to hold new words
    let split_text: Vec<String> = split(text);
    let mut split_bytes_ext: Vec<Vec<u32>> = 
        split_text
            .iter()
            .map(|s| s.as_bytes().iter().map(|&b| b as u32).collect())
            .collect();

    let mut pairs: HashMap<(u32, u32), u32> = HashMap::new();
    // for each loop.. merge 1 byte sequence and get 1 new word
    while new_word < target {
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
                .filter(|&(_, count)| *count >= 2)
                .max_by_key(|&(_, count)| count)
                .map(|(pair, &count)| (*pair, count)) {
            println!("found ({}, {}) {} times", byte1, byte2, count);

            // update Vocabulary
            vocab.vocab_vec.push(((byte1, byte2), new_word));
            vocab.vocab_hash.insert(new_word, (byte1, byte2));
            
            let string_view = format!("{}{}", vocab.stringify_word(&byte1), vocab.stringify_word(&byte1));
            println!("minting ({}, {}) into a new token {}", byte1, byte2, string_view);
            vocab.vocab_view.insert(new_word, string_view);
            
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
    }
    println!("extended vocabulary by {} from 256 words to {}", new_word-256, new_word); 
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

/// splits text into chunks and returns those in a vector
pub fn split(text: &str)  -> Vec<String> {
    // adaptation of pattern used for GPT-4 tokenizer
    let combined_pattern = format!(
        r"{}|{}|{}|{}|{}|{}",
        r"(?i:'(?:[sdmt]|ll|ve|re))", // contractions
        r"[^\r\n\p{L}\p{N}]\p{L}+",   // words
        r"\p{N}{1,3}",                // numbers
        r"[^\s\p{L}\p{N}]+[\r\n]*",   // special characters
        r"\s*[\r\n]",                 // newlines
        r"\s+"                        // spaces
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

    results
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