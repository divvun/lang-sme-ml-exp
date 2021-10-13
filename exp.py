# from aitextgen.tokenizers import train_tokenizer, TokenDataset, build_gpt2_config, aitextgen
from aitextgen import aitextgen
from aitextgen.colab import mount_gdrive, copy_file_from_gdrive
from aitextgen.TokenDataset import TokenDataset, merge_datasets
from aitextgen.utils import build_gpt2_config
from aitextgen.tokenizers import train_tokenizer
# train_tokenizer("./data/sme-freecorpus.txt")


# data = TokenDataset("./data/sme-freecorpus.txt", vocab_file=vocab_file, merges_file=merges_file, block_size=32)
config = build_gpt2_config(vocab_size=5000, max_length=32, dropout=0.0, n_embd=256, n_layer=8, n_head=8)
ai = aitextgen(tokenizer_file="aitextgen.tokenizer.json", config=config, to_gpu=True)
ai.train("./data/sme-freecorpus-small.txt", 
        line_by_line=False,
         from_cache=False,
         num_steps=100000,
         generate_every=1000,
         save_every=5,
        #  save_gdrive=False,
         learning_rate=1e-3,
         batch_size=8,
         fp16=True
         )