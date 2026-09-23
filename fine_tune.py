"""Fine-tune Laya's decision head on labeled POI dedup pairs, honest before/after.

Freezes the encoder + transformer head (the expensive general parts) and
fine-tunes only the ``scorer`` -- a 2-layer MLP that scores each level marker.
Because the encoder and head are frozen, we run them once, cache each level's
[MASK] embedding, and train the small scorer on those cached vectors. That is
the cheapest "route A" fine-tune and is feasible on CPU.

The dedup state already carries computed evidence (distance / name / brand /
category / phone match) as text, so what the fine-tune teaches the scorer is
"weigh that evidence" -- and it can read the *raw* distance, which the hand-set
feature_hint cannot (its distance term caps at 1.5 km).

    python fine_tune.py
"""
import json
import os
import shutil

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch
import torch.nn.functional as F

import laya
from laya.common import QTYPES, build_sequence, collate_items, confidence_from_probs

from app import config, decisions, features, models, overture, perturb

DATA = "overture_london.parquet"
FOCAL = {"lng_lo": -0.145, "lng_hi": -0.115, "lat_lo": 51.505, "lat_hi": 51.518}
K = 200          # labeled pairs per class (400 total)
EPOCHS = 20
LR = 1e-3
BATCH = 32
SEED = 0

# Soft target over the 5 dedup levels (0 = definitely different .. 4 = definitely same).
# Slightly soft so the scorer is not pushed to a hard one-hot and can still express
# lower confidence on hard cases.
T_SAME = [0.0, 0.0, 0.02, 0.13, 0.85]
T_DIFF = [0.85, 0.13, 0.02, 0.0, 0.0]


def focal_pool():
    pool = overture.load(DATA)
    return [p for p in pool
            if models.get(p.get("name")) and p["lat"] is not None
            and FOCAL["lng_lo"] <= p["lng"] <= FOCAL["lng_hi"]
            and FOCAL["lat_lo"] <= p["lat"] <= FOCAL["lat_hi"]]


def build_items(agent, pairs):
    tok = agent.tok
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    items = []
    for pair in pairs:
        f = features.pair_features(pair["poi_a"], pair["poi_b"])
        state = decisions.build_dedup_state_with_features(pair, f)
        q = agent._to_internal({
            "type": "score",
            "instructions": "How likely are POI A and POI B to be the same physical place?",
            "criteria": decisions.DEDUP_LEVELS,
        })
        seq, markers = build_sequence(tok, state, q, max_len, head_max_len)
        items.append({
            "ids": seq,
            "markers": markers,
            "qtype": QTYPES["score"],
            "target": T_SAME if pair["ground_truth"] else T_DIFF,
            "label": 1 if pair["ground_truth"] else 0,
            "pid": pair["id"],
        })
    return items


@torch.no_grad()
def cache_marker_embeddings(model, tok, items, batch=BATCH):
    """Run the frozen encoder+head once and cache each level's [MASK] embedding."""
    ms, masks = [], []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        b = collate_items([chunk], tok.pad_token_id)
        h = model.encoder(input_ids=b["input_ids"],
                          attention_mask=b["attention_mask"]).last_hidden_state
        h = h + model.type_emb(b["qtype"])[:, None, :]
        if model.head is not None:
            pad = ~b["attention_mask"].bool()
            for layer in model.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = b["marker_pos"].clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        ms.append(torch.gather(h, 1, idx))
        masks.append(b["marker_mask"])
    return torch.cat(ms), torch.cat(masks)


def logits_from(model, m, mask):
    logits = model.scorer(m).squeeze(-1).float()
    return logits.masked_fill(~mask, -1e4)


@torch.no_grad()
def predict(model, m, mask, temp=1.0):
    """score + confidence per example, replicating system_one's formula."""
    logits = logits_from(model, m, mask)
    out = []
    for i in range(logits.size(0)):
        k = int(mask[i].sum())
        z = logits[i, :k] / temp
        p = torch.exp(z - z.max())
        p = p / p.sum()
        s = float((torch.arange(k, dtype=p.dtype) * p).sum())
        conf = confidence_from_probs(p.cpu().numpy(), k)
        out.append((s, conf))
    return out


def report(name, preds, labels):
    acc = sum(1 for (s, _), l in zip(preds, labels) if (s >= 2) == bool(l)) / len(labels)
    tp = sum(1 for (s, _), l in zip(preds, labels) if s >= 2 and l == 1)
    fp = sum(1 for (s, _), l in zip(preds, labels) if s >= 2 and l == 0)
    fn = sum(1 for (s, _), l in zip(preds, labels) if s < 2 and l == 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    confs = [c for _, c in preds]
    auto = sum(1 for _, c in preds if c >= config.CONFIDENCE_AUTO)
    auto_ok = sum(1 for (s, c), l in zip(preds, labels)
                  if c >= config.CONFIDENCE_AUTO and (s >= 2) == bool(l))
    print(f"  {name:26s} no-gate acc={acc:.3f}  F1={f1:.3f}  "
          f"mean_conf={np.mean(confs):.3f}  auto={auto}/{len(labels)} (correct {auto_ok})")


def main():
    torch.manual_seed(SEED)
    agent = laya.load(config.LAYA_MODEL_ID, device="cpu", subfolder=config.LAYA_SUBFOLDER)
    model = agent.model
    print(f"loaded {config.LAYA_MODEL_ID}/{config.LAYA_SUBFOLDER} "
          f"(encoder {model.encoder.config.model_type}, {sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    pool = focal_pool()
    pairs = perturb.build_labeled_pairs(pool, k=K, seed=SEED)
    print(f"labeled pairs: {len(pairs)} "
          f"({sum(1 for p in pairs if p['ground_truth'])} pos / "
          f"{sum(1 for p in pairs if not p['ground_truth'])} neg)")

    items = build_items(agent, pairs)
    labels = torch.tensor([it["label"] for it in items])
    tgt = torch.tensor([it["target"] for it in items], dtype=torch.float32)

    print("caching marker embeddings (frozen encoder+head) ...")
    m_all, mask_all = cache_marker_embeddings(model, agent.tok, items)
    print(f"cached {m_all.shape}")

    # stratified 50/50 split
    rng = np.random.RandomState(SEED)
    pos = rng.permutation([i for i, p in enumerate(pairs) if p["ground_truth"]])
    neg = rng.permutation([i for i, p in enumerate(pairs) if not p["ground_truth"]])
    train_idx = np.concatenate([pos[:len(pos)//2], neg[:len(neg)//2]])
    hold_idx = np.concatenate([pos[len(pos)//2:], neg[len(neg)//2:]])
    rng.shuffle(train_idx); rng.shuffle(hold_idx)
    train_idx = torch.tensor(train_idx); hold_idx = torch.tensor(hold_idx)

    # ---- BEFORE: OOTB scorer on holdout ----
    print("\n--- before (OOTB scorer) ---")
    report("holdout", predict(model, m_all[hold_idx], mask_all[hold_idx]), labels[hold_idx])

    # ---- fine-tune the scorer only ----
    for p in model.parameters():
        p.requires_grad = False
    for p in model.scorer.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.scorer.parameters(), lr=LR)

    n_train = len(train_idx)
    print(f"\nfine-tuning scorer on {n_train} cached examples, {EPOCHS} epochs ...")
    for ep in range(EPOCHS):
        model.scorer.train()
        perm = torch.randperm(n_train)
        total = 0.0
        for i in range(0, n_train, BATCH):
            idx = train_idx[perm[i:i + BATCH]]
            logits = logits_from(model, m_all[idx], mask_all[idx])
            loss = -(tgt[idx] * F.log_softmax(logits, -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(idx)
        if ep % 4 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep:2d}  loss={total/n_train:.4f}")

    # ---- AFTER: fine-tuned scorer on holdout ----
    model.scorer.eval()
    print("\n--- after (fine-tuned scorer) ---")
    report("holdout", predict(model, m_all[hold_idx], mask_all[hold_idx]), labels[hold_idx])

    # hard-case spot check: same-brand different-branch negatives
    print("\n--- hard case: same-brand different-branch (should be 'different') ---")
    hard = [i for i in hold_idx.tolist() if pairs[i]["id"].startswith("neg_hard")][:4]
    for i in hard:
        f = features.pair_features(pairs[i]["poi_a"], pairs[i]["poi_b"])
        name = models.get(pairs[i]["poi_a"]["name"])
        d = f["distance_km"]
        s_b, c_b = predict(model, m_all[i:i+1], mask_all[i:i+1])[0]
        print(f"  {name!r}  dist={d:.3f}km  -> score={s_b:.2f} conf={c_b:.2f} "
              f"({'SAME' if s_b>=2 else 'DIFF'})")

    # ---- save as a loadable checkpoint ----
    out_dir = "laya_poi_finetuned"
    try:
        from huggingface_hub import snapshot_download
        from safetensors.torch import save_file
        prefix = f"{config.LAYA_SUBFOLDER}/" if config.LAYA_SUBFOLDER else ""
        base = snapshot_download(config.LAYA_MODEL_ID, allow_patterns=[
            prefix + n for n in ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")])
        src = os.path.join(base, config.LAYA_SUBFOLDER) if config.LAYA_SUBFOLDER else base
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        shutil.copytree(src, out_dir)          # dereferences the HF-cache symlinks
        save_file(model.state_dict(), os.path.join(out_dir, "model.safetensors"))
        print(f"\nsaved fine-tuned checkpoint to ./{out_dir}")
        # verify it loads back and produces a decision
        agent2 = laya.load(out_dir, device="cpu")
        ans = agent2.system_one(
            "POI A:\nname: BrewDog Soho\ncategory: bar\n\nPOI B:\nname: brewdog soho\ncategory: bar\n\ncomputed evidence:\n  distance_km: 0.06\n  name_similarity: 0.95",
            {"same_or_not": {"type": "score", "instructions": "same place?", "criteria": decisions.DEDUP_LEVELS}})
        a = ans["answers"]["same_or_not"]
        print(f"  reload check: score={a['score']:.2f} conf={a['confidence']:.2f}")
    except Exception as e:
        print(f"\n[note] checkpoint save skipped: {e!r}")


if __name__ == "__main__":
    main()
