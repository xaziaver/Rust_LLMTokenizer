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

    crate::tests::codepoint_test::execute();
    
}


/*
Things are complicated by the fact that we want to represent all kinds of
text beyond English (different languages, special characters, etc).

In the core language Rust has one string type, string slice (str) which are
references to some UTF-8 encoded string data. 

The String type, provided by Rust's standard library, however, is a
growable, mutable, owned, UTF-8 encoded string type.
*/