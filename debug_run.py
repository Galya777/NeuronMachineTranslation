import pickle
import numpy as np
import torch
import utils
import model
from parameters import *

print("🚀 DEBUG RUN STARTED")

# =========================================================
# 1. Подготовка на данни (ако още не са подготвени)
# =========================================================
try:
    trainCorpus, devCorpus = pickle.load(open(corpusFileName, 'rb'))
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    print("✅ Данните вече са подготвени")
except:
    print("📦 Подготвям данните...")
    trainCorpus, devCorpus, word2ind = utils.prepareData(
        sourceFileName, targetFileName,
        sourceDevFileName, targetDevFileName,
        '<S>', '</S>', '<UNK>', '<PAD>', '<TRANS>'
    )
    trainCorpus = [[word2ind.get(w, 2) for w in s] for s in trainCorpus]
    devCorpus = [[word2ind.get(w, 2) for w in s] for s in devCorpus]
    pickle.dump((trainCorpus, devCorpus), open(corpusFileName, 'wb'))
    pickle.dump(word2ind, open(wordsFileName, 'wb'))
    print("✅ Данните са записани")

# =========================================================
# 2. Създаване на модела
# =========================================================
print("🧠 Създавам модела...")

nmt = model.LanguageModel(
    vocab_size=len(word2ind),
    emb_dim=emb_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    start_idx=0,
    end_idx=1,
    pad_idx=3,
    trans_idx=4
).to(device)

optimizer = torch.optim.Adam(nmt.parameters(), lr=learning_rate)

# =========================================================
# 3. Малка тренировъчна стъпка (тест)
# =========================================================
print("🏋️ Тестова тренировка...")

nmt.train()
idx = np.arange(len(trainCorpus))
np.random.shuffle(idx)

batch = [trainCorpus[i] for i in idx[:batchSize]]
loss = nmt(batch)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("✅ Loss:", loss.item())

# =========================================================
# 4. Проверка на generate()
# =========================================================
print("🔮 Тестване на генериране...")

test_sentence = trainCorpus[0][:5]  # първите няколко токена
print("Input indices:", test_sentence)

nmt.eval()
with torch.no_grad():
    result = nmt.generate(test_sentence)

print("Generated indices:", result)

print("🎉 ВСИЧКО РАБОТИ")
