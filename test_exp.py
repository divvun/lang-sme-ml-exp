from aitextgen import aitextgen
# from aitextgen.utils import build_gpt2_config


# config = build_gpt2_config(vocab_size=5000, max_length=32, dropout=0.0, n_embd=256, n_layer=8, n_head=8)
ai=aitextgen(model_folder="./trained_model",
               tokenizer_file="aitextgen.tokenizer.json", 
               to_gpu=False)
ai.generate(n=5,
            batch_size=5,
            prompt="gaskab",
            temperature=1.0,
            top_p=0.9)