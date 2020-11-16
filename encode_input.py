#!/usr/bin/env python3

import re
import numpy as np

def clean_corpus(text):
    # clean very special char
    text = text.replace("¶", "").replace('•', '').replace('□', '').replace('§', '').replace('\uf03d', '').replace('π', '').replace('●', '').replace('µ', '').replace('º', '').replace('文', '').replace('中', '').replace('⅞', '').replace('½', '').replace('⅓', '').replace('¾', '').replace('¹', '').replace('³', '').replace('\t', '')
    # remove numbers
    text = re.sub(r'[0-9]+', '', text)
    # remove russian texts (it is in data)
    text = re.sub(r"[А-Яа-я]", '', text)
    # remove puctuation
    text = re.sub(r"[^\w\s]", "", text)

    return text

def encode_corpus():
    print("Reading free corpus...")
    with open('data/sme-freecorpus.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    text = clean_corpus(text)

    # encode the text
    # 1. int2char, integers to characters
    # 2. char2int, characters to unique integers
    chars = tuple(sorted(set(text)))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}

    # print(char2int)
    print("Encoding text...")

    # encode the text
    encoded = np.array([char2int[ch] for ch in text])

    with open("./encoded-corpus.bin", "wb") as f:
        np.save(f, encoded)

    with open("./corpus-chars.txt", "w") as f:
        f.write("".join(chars))

def load_encoded_corpus():
    return np.load('./encoded-corpus.bin')

def load_corpus_chars():
    with open("./corpus-chars.txt", "r") as f:
        return f.read()

if __name__ == "__main__":
    encode_corpus()
