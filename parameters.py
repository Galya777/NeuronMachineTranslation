import torch

sourceFileName = 'en_bg_data/train.bg'
targetFileName = 'en_bg_data/train.en'
sourceDevFileName = 'en_bg_data/dev.bg'
targetDevFileName = 'en_bg_data/dev.en'

corpusFileName = 'corpusData'
wordsFileName = 'wordsData'
modelFileName = 'NMTmodel'

device = torch.device("cuda:0")
#device = torch.device("cpu")

parameter1 = 1
parameter2 = 2
parameter3 = 3
parameter4 = 4

learning_rate = 0.001
batchSize = 32
clip_grad = 5.0

maxEpochs = 10
log_every = 10
test_every = 2000
