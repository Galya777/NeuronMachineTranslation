import torch

sourceFileName = 'en_bg_data/train.bg'
targetFileName = 'en_bg_data/train.en'
sourceDevFileName = 'en_bg_data/dev.bg'
targetDevFileName = 'en_bg_data/dev.en'

corpusFileName = 'corpusData'
wordsFileName = 'wordsData'
modelFileName = 'NMTmodel'

# Използвай GPU ако е наличен
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Хиперпараметри на модела
emb_dim = 256
hidden_dim = 512
num_layers = 1

learning_rate = 0.001
batchSize = 32
clip_grad = 5.0

maxEpochs = 10
log_every = 10
test_every = 2000
