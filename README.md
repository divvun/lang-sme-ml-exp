# lang-sme-ml-exp

## Requirements

You need Python 3.5+. Then you need to install the required python packages. Just run `pip3 install -r requirements.txt`.

If you don't have CUDA drivers installed and a compatible NVidia GPU, you need to to assign `--device=cpu` when training the model (script `train.py`).

## Training
dowload the data (not analyzed) and put in the `data` folder.

First, just run `encode_input.py` - this will create two files in your dir that will be used for training. This should be done only once.

You can start training the model with 

`./train.py --model-name {your model name} --lr 0.001 --epochs 15 --batch-size 128 --seq-len 300 --emb-dim=256 --is-gru --device={your device} --use-emb --n-hidden 756`

or change the paramaters (see `-h` for hepl when calling train.py). 

Note: `--resume-from-saved` now works as expected so you can stop and resume training whenever.

Note: you should have a `models` folder - it's where trained models will be saved to.

## Inference

To run inference `./predict.py models/{your model name}.pt --len 100 --first-word ja --device={your device}`   

The current inference functionality hasn't been tested yet. 




