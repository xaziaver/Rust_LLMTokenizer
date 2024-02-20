// gets a vector made of unicode code points as our method of "tokenization"
// this is not ideal since the vocabulary would be very long (~150k characters in unicode)
pub fn execute() {
    let test_string = "wave: \u{1F44B}, king: \u{2654}";
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
    println!("Input string: {}", test_string);
    println!("Output vector: {:?}", codepoint_vec);
    println!("##############################");
    println!("END CODE POINT TESTING");
    println!("");
}

// map out the starting byte indices of the characters
fn precompute_char_byte_indices(s: &str) -> Vec<usize> {
    s.char_indices().map(|(index, _)| index).collect()
}

// get the code point of a single character at the specified index 
// by using byte indices
fn get_unicode_codepoint(source: &str, index: usize, indices: &[usize]) -> Result<u32, &'static str> {
    
    let byte_index = *indices.get(index).ok_or("Index out of bounds")?;
    let next_byte_index = *indices.get(index + 1).unwrap_or(&source.len());
    let char_slice = &source[byte_index..next_byte_index];
    
    // safe unwrap, slice not empty
    let codepoint = char_slice.chars().next().unwrap() as u32;

    Ok(codepoint)
}