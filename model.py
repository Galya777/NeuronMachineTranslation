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
	def __init__(self, parameter1, parameter2, parameter3, parameter4):
		super(LanguageModel, self).__init__()
	
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
		return H
		
	def generate(self, prefix, limit=1000):
		return result
