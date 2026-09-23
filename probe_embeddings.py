"""Diagnostic: where (if anywhere) does the frozen Laya encoder carry a POI
dedup signal? Train a linear probe on three frozen representations and report
held-out accuracy. If none beats chance, a head-only fine-tune is hopeless and
the encoder itself must be trained.
"""
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch
import torch.nn.functional as F

import laya
from laya.common import QTYPES, build_sequence, collate_items

from app import config, decisions, features, models, overture, perturb

DATA = "overture_london.parquet"
FOCAL = {"lng_lo": -0.145, "lng_hi": -0.115, "lat_lo": 51.505, "lat_hi": 51.518}
K = 200
BATCH = 32
SEED = 0


def build_items(agent, pairs):
    tok = agent.tok
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    items = []
    for pair in pairs:
        f = features.pair_features(pair["poi_a"], pair["poi_b"])
        state = decisions.build_dedup_state_with_features(pair, f)
        q = agent._to_internal({"type": "score", "instructions": "same place?",
                                "criteria": decisions.DEDUP_LEVELS})
        seq, markers = build_sequence(tok, state, q, max_len, head_max_len)
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES["score"],
                      "label": 1 if pair["ground_truth"] else 0})
    return items


@torch.no_grad()
def cache(agent, items):
    model = agent.model
    tok = agent.tok
    enc_cls, head_cls, mask = [], [], []
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        b = collate_items([chunk], tok.pad_token_id)
        h_enc = model.encoder(input_ids=b["input_ids"],
                              attention_mask=b["attention_mask"]).last_hidden_state
        h = h_enc + model.type_emb(b["qtype"])[:, None, :]
        if model.head is not None:
            pad = ~b["attention_mask"].bool()
            for layer in model.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = b["marker_pos"].clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        enc_cls.append(h_enc[:, 0])
        head_cls.append(h[:, 0])
        mask.append(m.mean(dim=1))
    return torch.cat(enc_cls), torch.cat(head_cls), torch.cat(mask)


def probe(X, y, name, epochs=300, lr=1e-2):
    X = X.float()
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    n = len(y)
    rng = np.random.RandomState(SEED)
    pos = rng.permutation(np.where(y == 1)[0])
    neg = rng.permutation(np.where(y == 0)[0])
    tr = np.concatenate([pos[:len(pos)//2], neg[:len(neg)//2]])
    te = np.concatenate([pos[len(pos)//2:], neg[len(neg)//2:]])
    Xtr, ytr = X[tr], torch.tensor(y)[tr].float()
    Xte, yte = X[te], torch.tensor(y)[te].float()

    lin = torch.nn.Linear(X.size(1), 1)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lin(Xtr).squeeze(-1), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (lin(Xte).squeeze(-1) > 0).float()
    acc = (pred == yte).float().mean().item()
    print(f"  {name:16s} dim={X.size(1):5d}  held-out acc={acc:.3f}")
    return acc


def main():
    agent = laya.load(config.LAYA_MODEL_ID, device="cpu", subfolder=config.LAYA_SUBFOLDER)
    pool = overture.load(DATA)
    pool = [p for p in pool if models.get(p.get("name")) and p["lat"] is not None
            and FOCAL["lng_lo"] <= p["lng"] <= FOCAL["lng_hi"]
            and FOCAL["lat_lo"] <= p["lat"] <= FOCAL["lat_hi"]]
    pairs = perturb.build_labeled_pairs(pool, k=K, seed=SEED)
    items = build_items(agent, pairs)
    y = np.array([it["label"] for it in items])
    print(f"caching embeddings for {len(items)} pairs ...")
    enc_cls, head_cls, mask = cache(agent, items)
    print("linear probes (chance = 0.50):")
    probe(enc_cls, y, "encoder CLS")
    probe(head_cls, y, "head CLS")
    probe(mask, y, "mask mean")
    probe(torch.cat([enc_cls, head_cls], 1), y, "enc+head CLS")


if __name__ == "__main__":
    main()
