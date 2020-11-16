device = 'cuda:0'

import os
import os.path
import json

from baseline_preprocess_input import *
from models import *
from base_train import *

with open("./chars.txt") as f:
    chars = json.loads(f.read())
encoded = np.load("./encoded-corpus.bin")

n_hidden=512
n_layers=3
# default values
drop_prob = 0.5
lr=1
bidirectional = False

# load one of the models from models.py
model = LSTM(chars, device, bidirectional, n_hidden, n_layers)
print(model)

batch_size = 128
seq_length = 300
n_epochs = 1 # small because for testing

print("Training")

# train the model
resume_from_saved = os.path.exists("lstm3_epoch.pt")
train_and_save(model, encoded, device, model_name='lstm3_epoch.pt', epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.0001, resume_from_saved=resume_from_saved, bidirectional=bidirectional)

print("Adam")

opt = torch.optim.Adam(model.parameters(), lr=0.0001)

from predict import *

model, _ , _, _, _, _  = load_checkpoint("lstm3_epoch.pt", model, opt)

show_sample(model, 2000, device, prime='ja', top_k=5)

