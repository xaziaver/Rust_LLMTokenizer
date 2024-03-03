"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
from minbpe import BasicTokenizer, RegexTokenizer

# paths added for my project's context
train_text_path = "data/training_text.txt"
models_path = "python_tokenizer/models"
encode_input_path = "data/encode_text.txt"
encode_output_path = "data/output/encode_target.txt"

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

# adding encode tokens to file for comparison
encode_text = open(encode_input_path, "r", encoding="utf-8").read()
tokenizer = RegexTokenizer()
tokenizer.load(models_path + "/regex.model")
encode_result = tokenizer.encode_ordinary(encode_text)
with open(encode_output_path, "w") as f:
    for token in encode_result:
        f.write("%s\n" % token)