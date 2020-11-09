# lang-sme-ml-exp

## Requirements

You need Python 3.5+. Then you need to install the required python packages. Just run `pip3 install -r requirements.txt`.

If you don't have CUDA drivers installed and a compatible NVidia GPU, you need to manually change assigning of  `device` to `.cpu`.

## To run the baseline
dowload the data (not analyzed) and put in the `data` folder.

Just open the jupyter notebook and run from top.
To train over more epochs, change `n_epochs` before calling `train` function. 
`train` will produce print statements per epoch. 
`show_sample` will print generated text.

When you have a trained model, and you want to resume its training, change `resume_from_saved` to True. 



