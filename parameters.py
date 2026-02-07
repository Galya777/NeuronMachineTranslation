try:
    import torch
except ImportError:
    torch = None

sourceFileName = 'en_bg_data/train.bg'
targetFileName = 'en_bg_data/train.en'
sourceDevFileName = 'en_bg_data/dev.bg'
targetDevFileName = 'en_bg_data/dev.en'

corpusFileName = 'corpusData'
wordsFileName = 'wordsData'
modelFileName = 'NMTmodel'

# По подразбиране използваме CPU и правим параметъра съвместим, дори ако torch липсва
# Може да върнете CUDA, ако имате поддръжка: 'cuda:0' или torch.device("cuda:0")
device = ('cpu' if torch is None else torch.device("cpu"))

# Моделни хиперпараметри (по подразбиране)
emb_dim = 256
hidden_dim = 512
num_layers = 2

learning_rate = 0.001
batchSize = 32
clip_grad = 5.0

maxEpochs = 10
log_every = 10
test_every = 2000
