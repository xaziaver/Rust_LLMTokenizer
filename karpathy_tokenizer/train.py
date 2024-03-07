"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
from minbpe import BasicTokenizer, RegexTokenizer


###################################################################
from minbpe import render_token

# paths added for my project's context
models_path = "karpathy_tokenizer/models"
train_text_path = "data/train_text.txt"
train_output_path = "data/output/train_target.txt"
encode_input_path = "data/encode_text.txt"
encode_output_path = "data/output/encode_target.txt"
###################################################################


# open some text and train a vocab of 512 tokens
train_text = open(train_text_path, "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs(models_path, exist_ok=True)

t0 = time.time()
for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ["basic", "regex"]):

    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(train_text, 512, verbose=False)
    # writes two files in the models directory: name.model, and name.vocab
    prefix = os.path.join(models_path, name)
    tokenizer.save(prefix)
t1 = time.time()
print(f"Training took {t1 - t0:.2f} seconds")


###################################################################
# add encode tokens to file for comparison
encode_text = open(encode_input_path, "r", encoding="utf-8").read()
tokenizer = RegexTokenizer()
tokenizer.load(models_path + "/regex.model")
encode_result = tokenizer.encode_ordinary(encode_text)
# convert to readable characters
decode_result = tokenizer.decode(encode_result)
with open(encode_output_path, "w") as f:
    for token in decode_result:
        f.write("%s\n" % token.decode("utf-8"))

# add readable merges to file for comparison
# duplicating the logic in save() in base.py
with open(train_output_path, 'w') as f:
    inverted_merges = {idx: pair for pair, idx in tokenizer.merges.items()}
    for idx, token in tokenizer.vocab.items():
        if idx > 255:
            s = render_token(token)
            if idx in inverted_merges:
                idx0, idx1 = inverted_merges[idx]
                s0 = render_token(tokenizer.vocab[idx0])
                s1 = render_token(tokenizer.vocab[idx1])
                f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
            else:
                f.write(f"[{s}] {idx}\n")
###################################################################