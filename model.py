#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
#############################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import torch

class LanguageModel(torch.nn.Module):
	def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers,
	             start_idx=None, end_idx=None, pad_idx=None, trans_idx=None):
		super(LanguageModel, self).__init__()
		self.embedding = torch.nn.Embedding(vocab_size, emb_dim)
		self.gru = torch.nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.0 if num_layers==1 else 0.1)
		self.proj = torch.nn.Linear(hidden_dim, vocab_size)
		self.padTokenIdx = pad_idx if pad_idx is not None else 0
		self.startTokenIdx = start_idx if start_idx is not None else 0
		self.endTokenIdx = end_idx if end_idx is not None else 1
		self.transTokenIdx = trans_idx if trans_idx is not None else 4
		self._loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.padTokenIdx, reduction='sum')
	
	def preparePaddedBatch(self, source):
		device = next(self.parameters()).device
		m = max(len(s) for s in source)
		sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in source]
		return torch.tensor(sents_padded, dtype=torch.long, device=device)	# shape=(batch_size, seq_len)

	def save(self,fileName):
		torch.save(self.state_dict(), fileName)

	def load(self,fileName):
		self.load_state_dict(torch.load(fileName))

	def forward(self, source):
		# source: list of token id lists
		X = self.preparePaddedBatch(source)  # (B, T)
		if X.size(1) < 2:
			return torch.tensor(0.0, device=X.device)
		inputs = X[:, :-1]
		targets = X[:, 1:]
		emb = self.embedding(inputs)  # (B, T-1, E)
		h, _ = self.gru(emb)  # (B, T-1, H)
		logits = self.proj(h)  # (B, T-1, V)
		B, Tm1, V = logits.shape
		logits = logits.reshape(B*Tm1, V)
		targets = targets.reshape(B*Tm1)
		loss_sum = self._loss_fn(logits, targets)
		valid = (targets != self.padTokenIdx).sum().clamp(min=1)
		H = loss_sum / valid
		return H
		
	def generate(self, prefix, limit=1000):
		# prefix: list of token ids
		device = next(self.parameters()).device
		seq = list(prefix)
		# Initialize hidden with running the prefix through the GRU
		with torch.no_grad():
			X = torch.tensor([seq], dtype=torch.long, device=device)
			emb = self.embedding(X)
			out, h = self.gru(emb)
			while len(seq) < limit:
				# take last token as input
				last_tok = torch.tensor([[seq[-1]]], dtype=torch.long, device=device)
				emb_last = self.embedding(last_tok)
				out, h = self.gru(emb_last, h)
				logits = self.proj(out[:, -1, :])  # (1, V)
				next_tok = int(torch.argmax(logits, dim=-1).item())
				seq.append(next_tok)
				if next_tok == self.endTokenIdx:
					break
		return seq
