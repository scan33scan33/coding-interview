import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Shape suffix convention:
#   B = batch, L = sequence length, D = model dim
# A tensor's name tells you its shape. If an op's output name doesn't
# follow from its input names, the op is probably wrong.


class BiEncoder(nn.Module):
    def __init__(self, backbone, temp=0.05):
        super().__init__()
        self.backbone = backbone
        self.log_temp = nn.Parameter(torch.tensor(temp).log())

    def forward(self, input_ids, attention_mask):
        h_BLD = self.backbone(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        mask_BL1 = attention_mask.unsqueeze(-1).float()
        pooled_BD = (h_BLD * mask_BL1).sum(1) / mask_BL1.sum(1).clamp(min=1)
        return F.normalize(pooled_BD, dim=-1)


def info_nce(q_BD, d_BD, doc_id_B, log_temp):
    B = q_BD.size(0)
    sim_BB = q_BD @ d_BD.T / log_temp.exp()
    dup_BB = doc_id_B[:, None] == doc_id_B[None, :]
    dup_BB &= ~torch.eye(B, dtype=torch.bool, device=sim_BB.device)
    sim_BB = sim_BB.masked_fill(dup_BB, float("-inf"))
    return -F.log_softmax(sim_BB, dim=-1).diagonal().mean()


class PairDataset(Dataset):
    def __init__(self, queries, docs):
        self.queries, self.docs = queries, docs

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        return self.queries[i], self.docs[i]


def make_collate(tok):
    def collate(batch):
        queries, docs = zip(*batch)
        q = tok(list(queries), padding=True, truncation=True, return_tensors="pt")
        d = tok(list(docs), padding=True, truncation=True, return_tensors="pt")
        doc_id_B = torch.tensor([hash(t) for t in docs])
        return q, d, doc_id_B
    return collate


def train(model, loader, steps, accum=8, lr=2e-5, device="cuda"):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    opt.zero_grad()
    for i, (q, d, doc_id_B) in enumerate(loader):
        q, d = {k: v.to(device) for k, v in q.items()}, {k: v.to(device) for k, v in d.items()}
        loss = info_nce(model(**q), model(**d), doc_id_B.to(device), model.log_temp)
        (loss / accum).backward()
        if (i + 1) % accum == 0:
            opt.step()
            opt.zero_grad()
        if i % 100 == 0:
            print(f"step {i}  loss {loss.item():.4f}")
        if i >= steps:
            break
