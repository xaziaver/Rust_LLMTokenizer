// gets a vector made of the bytes of a string encoded with UTF-8
// if we used just this, then our vocabulary length is at most 256
pub fn execute(test_string: &str) -> Vec<u32> {

    // Rust stores strings as UTF-8 encoded bytes
    let utf8_vec: Vec<u8> = test_string.as_bytes().to_vec();
    utf8_vec.into_iter().map(|val| val as u32).collect()
}
// a short vocabulary length means input text gets stretched across very long
// sequences of bytes... the embedding tables and prediction at the final layer
// will both be tiny, but this will cause a problem since we have a finite context
// length in the attention that can be supported in a transformer

/* ^^^ NOTE ^^^ 
There is actually a huge improvement that could be made if we could feed raw byte 
sequences into LLMs because this would eliminate tokenization. However, the
transformer architecture would need to be modified because of the problem above.

In a recent paper, some have proposed a hierarchical structuring of the 
transformer to allow the improvement
*/
