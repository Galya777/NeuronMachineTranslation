#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
#############################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModel(nn.Module):
    """
    Езиков модел върху конкатенация: <S> source ... <TRANS> target ... </S>
    - Връща средна по дума негативна лог-вероятност (натурален лог) за batch.
    - generate: greedy decode, започвайки от подадения prefix (списък от индекси),
      докато стигне </S> или лимит.
    """

    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers,
                 start_idx, end_idx, pad_idx, trans_idx):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, vocab_size)

        self.startTokenIdx = start_idx
        self.endTokenIdx = end_idx
        self.padTokenIdx = pad_idx
        self.transTokenIdx = trans_idx

    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents_padded = [s + (m - len(s)) * [self.padTokenIdx] for s in source]
        return torch.tensor(sents_padded, dtype=torch.long, device=device)  # (B, T)

    def save(self, fileName):
        torch.save(self.state_dict(), fileName)

    def load(self, fileName):
        self.load_state_dict(torch.load(fileName, map_location=next(self.parameters()).device))

    def forward(self, source):
        """
        source: list от списъци (индекси). Връща средна per-token NLL (scalar тензор).
        """
        x = self.preparePaddedBatch(source)  # (B, T)
        B, T = x.shape
        # Вход до RNN: всички освен последния токен, предсказваме следващия
        inp = x[:, :-1]
        tgt = x[:, 1:]

        mask = (tgt != self.padTokenIdx)  # (B, T-1)

        emb = self.emb(inp)  # (B, T-1, E)
        h, _ = self.rnn(emb)  # (B, T-1, H)
        logits = self.proj(h)  # (B, T-1, V)
        log_probs = F.log_softmax(logits, dim=-1)

        # Съберем log_prob на истинските таргети
        tgt_flat = tgt.reshape(-1)
        lp_flat = log_probs.reshape(-1, self.vocab_size)        # (B*(T-1), V)
        token_lp = lp_flat[torch.arange(tgt_flat.numel(), device=lp_flat.device), tgt_flat]  # (B*(T-1))
        token_lp = token_lp.view(B, T - 1)

        # Вземаме само валидните позиции (не PAD), средно по валидни токени
        valid_counts = mask.sum().clamp(min=1)
        H = - (token_lp[mask].sum() / valid_counts)  # average NLL per token
        return H

    @torch.no_grad()
    def generate(self, prefix, limit=1000):
        """
        prefix: списък от индекси. Връща цялата последователност (prefix + генерирано).
        """
        device = next(self.parameters()).device
        seq = list(prefix)
        # Подготвяме начално състояние чрез пускане на prefix през RNN
        x = torch.tensor([seq], dtype=torch.long, device=device)  # (1, T)
        emb = self.emb(x)
        h, state = self.rnn(emb)

        cur = seq[-1] if len(seq) > 0 else self.startTokenIdx
        steps = 0
        while steps < limit and cur != self.endTokenIdx:
            # последното изходно скрито състояние -> проекция към логити
            logits = self.proj(h[:, -1:, :])  # (1,1,V)
            next_token = torch.argmax(logits.squeeze(0).squeeze(0), dim=-1).item()
            seq.append(next_token)
            cur = next_token
            # подай следващия като вход
            x_next = torch.tensor([[cur]], dtype=torch.long, device=device)
            emb_next = self.emb(x_next)
            h, state = self.rnn(emb_next, state)
            steps += 1
            if cur == self.endTokenIdx:
                break
        return seq
