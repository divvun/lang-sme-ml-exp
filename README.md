# lang-sme-ml-exp

## Requirements

You need Python 3.5+. Then you need to install the required python packages. Just run `pip3 install -r requirements.txt`.

If you don't have CUDA drivers installed and a compatible NVidia GPU, you need to to assign `--device=cpu` when training the model (script `train.py`).

## Training
dowload the data (dependency analysis) and put in the `data` folder.

First, just run `encode_words.py` - this will create two files in your dir that will be used for training. This should be done only once.

You can start training the model with 

New args for AWS env:

--model-dir (for AWS env) the path to folder with `--model-name`.

--train-dir - where the training (preprocessed) data is, `train-words-enc.csv`

--val-dir - validation data `val-words-enc.csv`

--vocab - dictionary int2word.txt (not used yet and commented)

--output-dir - commented 

`./train.py start --device=cuda --model-name {your model name} --epochs 3 --batch-size 64 --seq-len 2 --use-emb`

or with changed paramaters (see `-h` for help when calling train.py). 

Resume training with 

`./train.py resume --device=cuda:0 --model-name {your model name} --epochs 5` 

Note: you should have a `models` folder - it's where trained models will be saved to.

Note: adding `--bidirectional`, shrinking `--lr`, increasing `--batch-size`, `--n-hidden` and `--n-layers` would make the training time longer. However, this doesn't influence inference time. 

## Inference

To run inference `./predict.py models/{your model name}.pt --n-preds 3 --first-word 'ja' --device={your device}`, where `n-preds` is the number of scenarios you want to see predicted.

If you don't type a white space after `--first-word`, it will try to autocomplete the input and only then give a prediction per scenario. 
If the white space is a part of input, it will be considered a full word and prediction will start after it. 

