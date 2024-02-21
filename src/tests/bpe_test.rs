use std::collections::HashMap;

// instead of using the raw UTF-8 bytes we want to support a larger vocabulary size
// that we can tune as a hyperparameter while sticking with the same encoding
pub fn execute(test_string: &str) -> Vec<u32> {

    let increase_vocab_size = 1;
    let bpe_vec: Vec<u8> = byte_pair_encoding(&test_string.as_bytes().to_vec(), increase_vocab_size);
    bpe_vec.into_iter().map(|val| val as u32).collect()
}

// the BPE algorithm allows us to compress our sequence of bytes
// https://en.wikipedia.org/wiki/Byte_pair_encoding
fn byte_pair_encoding(utf8_bytes: &Vec<u8>, increase: u32) -> Vec<u8> {
    
    // idea is that we iteratively compress our byte sequence
    // while "minting" new tokens (expanding the vocabulary)

    // we have a vocabulary size of 256 with bytes
    // we find the most frequent byte-pair and mint as a new token

    // x tokens with vocabulary of y becomes
    // x - i tokens with a vocabulary of y + j


    /* let mut compressed_bytes: Vec<u8> = vec!();
    let mut lib_len = 256;
    let mut pair_occurs: HashMap<(u8, u8), u32> = HashMap::new(); */

    utf8_bytes.clone()
}