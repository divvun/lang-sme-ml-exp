# lang-sme-ml-exp

## Requirements

You need Python 3.5+. Then you need to install the required python packages. Just run `pip3 install -r requirements.txt`.

If you don't have CUDA drivers installed and a compatible NVidia GPU, you need to to assign `--device=cpu` when training the model (script `train.py`).

## Training
dowload the data (not analyzed) and put in the `data` folder.

First, just run `encode_input.py` - this will create two files in your dir that will be used for training. This should be done only once.

You can start training the model with 

`./train.py start --device=cuda:0 --model-name {your model name} --epochs 1 --batch-size 128 --seq-len 100`

or with changed paramaters (see `-h` for help when calling train.py). 

Resume training with 

`./train.py resume --device=cuda:0 --model-name {your model name} --epochs 5` 

Note: `resume` now works as expected so you can stop and resume training whenever.

Note: you should have a `models` folder - it's where trained models will be saved to.

Note: adding `--bidirectional`, shrinking `--lr`, increasing `--batch-size`, `--n-hidden` and `--n-layers` would make the training time longer. However, this doesn't influence inference time. 

## Inference

To run inference `./predict.py models/{your model name}.pt --n_words 3 --first-word ja --device={your device}`, where `n-words` is the number of words you want to see predicted.

If you want to run `predict.py` with more then one input word, please, give them with '' - `--first-word 'on the'`. 

All combinations of all hyperparamaters were tested and they work. As for the performance, LSTM (default) option gives better results than GRU in terms of prediction of real words. `--use-emb` and `--bidirectional` options work, but it wasn't yet established whether they improve the performance. 

The best model wasn't established so far, but LSTM was giving me 96% accuracy after 5 epochs with 1000 chars of output.
Note: accuracy in this case is just a proportion between real_words_predicted/all_words_predicted. 