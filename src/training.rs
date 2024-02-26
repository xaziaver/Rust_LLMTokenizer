use std::collections::HashMap;

pub struct Vocabulary {
    pub vocab_hash: HashMap<u32, (u32, u32)>,    // for decoding
    pub vocab_vec: Vec<((u32, u32), u32)>,      // for encoding
}

// instead of using the raw UTF-8 bytes we want to support a larger vocabulary size
// that we can tune as a hyperparameter while sticking with the same encoding
pub fn execute(test_string: &str) -> Vocabulary {
    let target_vocab_size = 500;
    let new_vocabulary = train_tokenizer(test_string.as_bytes().to_vec(), target_vocab_size);
    new_vocabulary
}


// TODO: improve new token creation w/ GPT2 method: prevent seperate merges for (e.g. dog, dog., dog!, dog?, ..)
//       by splitting input text into groups, merging on those groups and taking the union for final Vocabulary.

// TODO: look at available inference source code (https://github.com/openai/tiktoken) to try to reverse
// engineer training methods... tiktoken/tiktoken_ext/openai_public.py shows some details
// tiktokenizer.vercel.app

// TODO: incorporate special tokens after researching & determining what's needed
//       (requires changes to transformer's code to account for new special tokens)

// the BPE algorithm: compresses `utf8_bytes` while increasing vocabulary to `target`
// https://en.wikipedia.org/wiki/Byte_pair_encoding
fn train_tokenizer(utf8_bytes: Vec<u8>, target: u32) -> Vocabulary {
    let mut vocab = Vocabulary {
        vocab_hash: HashMap::<u32, (u32, u32)>::new(),
        vocab_vec: Vec::<((u32, u32), u32)>::new(),
    };
    let mut extended_bytes: Vec<u32> = utf8_bytes.into_iter().map(|val| val as u32).collect();
    let mut new_word: u32 = 256;    // start with 256 as the first 'new' word after the initial byte range

    for _ in 0..target-256 {
        if let Some(((byte1, byte2), count)) = 
            // find the pair with the maximum count (at least 2)
            pair_counts(&extended_bytes).into_iter()
                .filter(|&(_, count)| count >= 2)
                .max_by_key(|&(_, count)| count) {
            println!("found ({}, {}) {} times", byte1, byte2, count);
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