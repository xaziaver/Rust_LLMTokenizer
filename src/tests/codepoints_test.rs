// gets a vector made of the Unicode code points of a string as our method of "tokenization"
// this is not ideal since the vocabulary would be very long (~150k characters in unicode)
pub fn execute(test_string: &str) -> Vec<u32> {
    let indices = precompute_char_byte_indices(&test_string);

    let mut codepoint_vec: Vec<u32> = vec![];
    println!("START CODE POINT TESTING");
    println!("##############################");
    for i in 0..indices.len() {
        match get_unicode_codepoint(&test_string, i, &indices) {
            Ok(codepoint) => codepoint_vec.push(codepoint),
            Err(e) => {
                println!("Error: {}", e);
                break;
            },
        }
    }
    codepoint_vec
}

/// Maps out the starting byte indices of the characters in a string.
fn precompute_char_byte_indices(s: &str) -> Vec<usize> {
    s.char_indices().map(|(index, _)| index).collect()
}

/// Gets the Unicode code point of the character at the specified index using byte indices.
fn get_unicode_codepoint(source: &str, index: usize, indices: &[usize]) -> Result<u32, &'static str> {
    
    let source_len = source.len();
    let byte_index = indices.get(index).ok_or("Index out of bounds")?;
    let next_byte_index = indices.get(index + 1).unwrap_or(&source_len);
    let char_slice = &source[*byte_index..*next_byte_index];
    
    // map each character to a u32 to get the code point
    char_slice.chars().next()
        .map(|ch| ch as u32)
        .ok_or("Unexpected error: character data could not be accessed")
}