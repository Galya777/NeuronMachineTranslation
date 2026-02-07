#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
#############################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import sys
import numpy as np
import math
import pickle
import time

import utils
from parameters import *

startToken = '<S>'
startTokenIdx = 0

endToken = '</S>'
endTokenIdx = 1

unkToken = '<UNK>'
unkTokenIdx = 2

padToken = '<PAD>'
padTokenIdx = 3

transToken = '<TRANS>'
transTokenIdx = 4

def perplexity(nmt, test, batchSize):
    import importlib
    torch = importlib.import_module('torch')
    testSize = len(test)
    H = 0.
    c = 0
    for b in range(0,testSize,batchSize):
        batch = test[b:min(b+batchSize, testSize)]
        l = sum(len(s)-1 for s in batch)
        c += l
        with torch.no_grad():
            H += l * nmt(batch)
    return math.exp(H/c)

if len(sys.argv)>1 and sys.argv[1] == 'prepare':
    trainCorpus, devCorpus, word2ind = utils.prepareData(sourceFileName, targetFileName, sourceDevFileName, targetDevFileName, startToken, endToken, unkToken, padToken, transToken)
    trainCorpus = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in trainCorpus ]
    devCorpus = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in devCorpus ]
    pickle.dump((trainCorpus, devCorpus), open(corpusFileName, 'wb'))
    pickle.dump(word2ind, open(wordsFileName, 'wb'))
    print('Data prepared.')

if len(sys.argv)>1 and (sys.argv[1] == 'train' or sys.argv[1] == 'extratrain'):
    import importlib
    torch = importlib.import_module('torch')
    model = importlib.import_module('model')
    (trainCorpus,devCorpus) = pickle.load(open(corpusFileName, 'rb'))
    word2ind = pickle.load(open(wordsFileName, 'rb'))

    nmt = model.LanguageModel(
        vocab_size=len(word2ind),
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        start_idx=startTokenIdx,
        end_idx=endTokenIdx,
        pad_idx=padTokenIdx,
        trans_idx=transTokenIdx
    ).to(device)
    optimizer = torch.optim.Adam(nmt.parameters(), lr=learning_rate)

    if sys.argv[1] == 'extratrain':
        nmt.load(modelFileName)
        (iter,bestPerplexity,learning_rate,osd) = torch.load(modelFileName + '.optim')
        optimizer.load_state_dict(osd)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    else:
        bestPerplexity = math.inf
        iter = 0

    idx = np.arange(len(trainCorpus), dtype='int32')
    nmt.train()
    beginTime = time.time()
    for epoch in range(maxEpochs):
        np.random.shuffle(idx)
        words = 0
        trainTime = time.time()
        for b in range(0, len(idx), batchSize):
			#############################################################################
			### Може да се наложи да се променя скоростта на спускане learning_rate в зависимост от итерацията
			#############################################################################
            iter += 1
            batch = [ trainCorpus[i] for i in idx[b:min(b+batchSize, len(idx))] ]
            
            words += sum( len(s)-1 for s in batch )
            H = nmt(batch)
            optimizer.zero_grad()
            H.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(nmt.parameters(), clip_grad)
            optimizer.step()
            if iter % log_every == 0:
                print("Iteration:",iter,"Epoch:",epoch+1,'/',maxEpochs,", Batch:",b//batchSize+1, '/', len(idx) // batchSize+1, ", loss: ",H.item(), "words/sec:",words / (time.time() - trainTime), "time elapsed:", (time.time() - beginTime) )
                trainTime = time.time()
                words = 0
                
            if iter % test_every == 0:
                nmt.eval()
                currentPerplexity = perplexity(nmt, devCorpus, batchSize)
                nmt.train()
                print('Current model perplexity: ',currentPerplexity)

                if currentPerplexity < bestPerplexity:
                    bestPerplexity = currentPerplexity
                    print('Saving new best model.')
                    nmt.save(modelFileName)
                    torch.save((iter,bestPerplexity,learning_rate,optimizer.state_dict()), modelFileName + '.optim')

    print('reached maximum number of epochs!')
    nmt.eval()
    currentPerplexity = perplexity(nmt, devCorpus, batchSize)
    print('Last model perplexity: ',currentPerplexity)
        
    if currentPerplexity < bestPerplexity:
        bestPerplexity = currentPerplexity
        print('Saving last model.')
        nmt.save(modelFileName)
        torch.save((iter,bestPerplexity,learning_rate,optimizer.state_dict()), modelFileName + '.optim')

if len(sys.argv)>3 and sys.argv[1] == 'perplexity':
    import importlib
    torch = importlib.import_module('torch')
    model = importlib.import_module('model')
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    
    nmt = model.LanguageModel(
        vocab_size=len(word2ind),
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        start_idx=startTokenIdx,
        end_idx=endTokenIdx,
        pad_idx=padTokenIdx,
        trans_idx=transTokenIdx
    ).to(device)
    nmt.load(modelFileName)
    
    sourceTest = utils.readCorpus(sys.argv[2])
    targetTest = utils.readCorpus(sys.argv[3])
    test = [ [startToken] + s + [transToken] + t + [endToken] for (s,t) in zip(sourceTest,targetTest)]
    test = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in test ]

    nmt.eval()
    print('Model perplexity: ', perplexity(nmt, test, batchSize))

if len(sys.argv)>3 and sys.argv[1] == 'translate':
    import importlib
    torch = importlib.import_module('torch')
    model = importlib.import_module('model')
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    words = list(word2ind)

    sourceTest = utils.readCorpus(sys.argv[2])
    test = [ [startToken] + s + [transToken] for s in sourceTest ]
    test = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in test ]

    nmt = model.LanguageModel(
        vocab_size=len(word2ind),
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        start_idx=startTokenIdx,
        end_idx=endTokenIdx,
        pad_idx=padTokenIdx,
        trans_idx=transTokenIdx
    ).to(device)
    nmt.load(modelFileName)

    nmt.eval()
    file = open(sys.argv[3],'w')
    pb = utils.progressBar()
    pb.start(len(test))
    for s in test:
        r=nmt.generate(s)
        st = r.index(transTokenIdx)
        result = [words[i] for i in r[st+1:-1]]
        file.write(' '.join(result)+"\n")
        pb.tick()
    pb.stop()

if len(sys.argv)>2 and sys.argv[1] == 'generate':
    import importlib
    torch = importlib.import_module('torch')
    model = importlib.import_module('model')
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    words = list(word2ind)

    test = sys.argv[2].split()
    test = [word2ind.get(w,unkTokenIdx) for w in test]

    nmt = model.LanguageModel(
        vocab_size=len(word2ind),
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        start_idx=startTokenIdx,
        end_idx=endTokenIdx,
        pad_idx=padTokenIdx,
        trans_idx=transTokenIdx
    ).to(device)
    nmt.load(modelFileName)

    nmt.eval()
    r=nmt.generate(test)
    result = [words[i] for i in r]
    print(' '.join(result)+"\n")

if len(sys.argv)>3 and sys.argv[1] == 'bleu':
    # Динамичен импорт, за да избегнем твърда зависимост ако BLEU не се ползва
    import importlib
    try:
        bleu_mod = importlib.import_module('nltk.translate.bleu_score')
        corpus_bleu = getattr(bleu_mod, 'corpus_bleu')
    except ImportError as e:
        print("NLTK не е инсталиран. Моля, инсталирайте го с 'pip install nltk' за да ползвате BLEU.")
        sys.exit(1)
    ref = [[s] for s in utils.readCorpus(sys.argv[2])]
    hyp = utils.readCorpus(sys.argv[3])

    bleu_score = corpus_bleu(ref, hyp)
    print('Corpus BLEU: ', (bleu_score * 100))
