import torch

sourceFileName = 'akkadian_data/train.akk'
targetFileName = 'akkadian_data/train.en'
sourceDevFileName = 'akkadian_data/dev.akk'
targetDevFileName = 'akkadian_data/dev.en'

corpusFileName = 'akkadian_corpusData'
wordsFileName = 'akkadian_wordsData'
modelFileName = 'AkkadianNMTmodel'

# Използвай GPU ако е наличен
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Хиперпараметри на модела
emb_dim = 128
hidden_dim = 256
num_layers = 1 # Simplified for low memory

learning_rate = 0.0005 # Lower learning rate
batchSize = 16 # Decreased batch size
clip_grad = 5.0

maxEpochs = 20 # Reduced epochs for faster submission
log_every = 10
test_every = 50
