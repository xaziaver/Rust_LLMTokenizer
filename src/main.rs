/*
Q: What do we want to do?
A: take strings and feed them into language models

For that we need to
1. "tokenize" the strings into a set of integers in some fixed vocabulary
2. those integers will be used to make a lookup into a table of vectors
3. the result vectors will be fed into the transformer as an input
*/

pub mod tests;

fn main() {

    let to_be_tokenized: &str = "wave: \u{1F44B}, king: \u{2654}";

    let test_cases = vec![
        ("CODEPOINTS", Box::new(crate::tests::codepoints_test::execute) as Box<dyn Fn(&str) -> Vec<u32>>),
        ("UTF8 BYTES", Box::new(crate::tests::utf8_bytes_test::execute)),
        ("BYTE PAIR ENCODING", Box::new(crate::tests::bpe_test::execute)),
    ];

    for (test_name, test_fn) in test_cases {
        println!("START {} TESTING", test_name);
        println!("##############################");
        println!("Input string: {}", to_be_tokenized);
        
        let result_vec = test_fn(to_be_tokenized);
        println!("Output vector: {:?}", result_vec);
        
        println!("##############################");
        println!("END {} TESTING", test_name);
        println!("");
    }
}


/*
Things are complicated by the fact that we want to represent all kinds of
text beyond English (different languages, special characters, etc).

In the core language Rust has one string type, string slice (str) which are
references to some UTF-8 encoded string data. 

The String type, provided by Rust's standard library, however, is a
growable, mutable, owned, UTF-8 encoded string type.
*/