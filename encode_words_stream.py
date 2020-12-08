import numpy as np
import re
import csv
from collections import Counter
# import pandas as pd

from parser import parse_file

def create_lookup_tables(words):
    word_counts = Counter(words)
    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # create int_to_vocab dictionaries
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

# def ignore_block(block_items):
#     is_num = re.search('[\d]+.*>', block_items[0])
#     if is_num is not None:
#         return True

#     is_alpha = re.search('[^\w\s]>', block_items[0])
#     if is_alpha is not None:
#         return True

#     return False

# def strip_whitespace(text):
#     text = text.lstrip()
#     text = text.rstrip()
#     return text


# def find_tokens(text):
#     blocks = text.replace('\\n', '').split('"<') 
#     words = []
#     for block in blocks:
#         block = strip_whitespace(block)
#         items = block.split('\n')

#         if ignore_block(items):
#             continue

#         if len(items)<2:
#             continue

#         word_matches = re.search(r"(.*?)>", strip_whitespace(items[0]))
  
#         word_pattern = '\".+\"'
#         pos_pattern = "([\w\?/\\-á]+)"

#         pos_matches = re.search(f"{word_pattern}\s+{pos_pattern}\s?", strip_whitespace(items[1]))
#         if pos_matches is None:
#             pos_matches = re.search(f"{word_pattern}\s+<\w+>\s+{pos_pattern}\s?", strip_whitespace(items[1]))
#         if pos_matches is None:
#             pos_matches = re.search(f"{word_pattern}\s+<\w+>\s+<\w+>\s+{pos_pattern}\s?", strip_whitespace(items[1]))
#         if pos_matches is None:
#             pos_matches = re.search(f"{word_pattern}\s+<\w+>\s+<\w+>\s+<\w+>\s+{pos_pattern}\s?", strip_whitespace(items[1]))

#         word = word_matches.group(1)
#         pos = pos_matches.group(1)

#         if pos == '?':
#             continue

#         words.append((word, pos))

#     return words

def encode_corpus():
    print('Reading analyzed corpus...')
    fn = 'data/sme-freecorpus-dependency-analysis.txt'

    # with open('data/sme-freecorpus-dependency-analysis.txt', 'r', encoding='utf-8') as f:
        # text = chars(f)

    words_and_pos = parse_file(fn)

    words_only = (x["name"] for x in words_and_pos)

    # words_only = [w[0] for w in words_and_pos]
    # pos_only = [w[1] for w in words_and_pos] 

    # print(words_only[:100])  
    # print(pos_only[:100])

    # with open('./all_words.txt', 'w') as f:
    #     f.write('\n'.join([w for w in words_only]))
    word2int, int2word = create_lookup_tables(words_only)
    # int2word = dict(enumerate(set(words_only)))
    # word2int = {w:ii for ii, w in int2word.items()}

    int2pos = dict(enumerate(set(pos_only)))
    pos2int = {p:ii for ii, p in int2pos.items()}

    print('Found words: ', len(word2int))
    print("Found pos: ", len(pos2int))

    print("Encoding text ...")

    # words = list(word2int.keys())
    # print(words[:10])

    encoded_w = [word2int[w] for w in words_only]

    val_idx = int(len(encoded_w)*(1-0.1))

    encoded_train = encoded_w[:val_idx]
    train_enc_x = encoded_train[:len(encoded_train)-1]
    train_enc_y = encoded_train[1:]

    encoded_val = encoded_w[val_idx:]
    val_enc_x = encoded_val[:len(encoded_val)-1]
    val_enc_y = encoded_val[1:]
    encoded_p = np.array(list(int2pos.keys()), dtype=int)

    with open('./train_words_enc.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(zip(train_enc_x, train_enc_y))

    with open('./val_words_enc.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(zip(val_enc_x, val_enc_y))

    # with open('./val.csv', 'w', newline='') as f: 
        # np.savetxt(f, train_y, fmt='%i')

    # with open('./encoded-words-val.csv', 'w', newline='') as f:
    #     np.savetxt(f, encoded_val, fmt='%i') 
        
    # with open('./encoded-pos.bin', 'wb') as f:    
    #     np.save(f, encoded_p)

    with open('./corpus-words.txt', 'w') as f:
        f.write('\n'.join(word2int.keys()))

def load_whole_corpus():
    with open('train_words_enc.csv', 'r') as f:
        train = list(csv.reader(f))
        X_train = [int(i[0]) for i in train]
        y_train = [int(i[1]) for i in train]
    with open('val_words_enc.csv', 'r') as f:
        val = list(csv.reader(f))
        X_val = [int(i[0]) for i in val]
        y_val = [int(i[1]) for i in val]
    return X_train, y_train, X_val, y_train
       
# def load_pos():
#     with open('./encoded-pos.txt', 'r') as f:
#         return f.read

def load_corpus_words():
    with open('./corpus-words.txt', 'r') as f:
        return f.read().split('\n')

def buffer_as_list(iter, bytes=128):
    x = []
    try:
        while True:
            for _ in range(0, bytes):
                x.append(next(iter))
            yield x
            x = []
    except StopIteration:
        yield x

if __name__ == '__main__':
    encode_corpus()
