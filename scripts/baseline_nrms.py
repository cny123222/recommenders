"""
Baseline NRMS (GloVe) — Pure PyTorch, self-contained.
Quick pipeline verification: train → eval → prediction.zip
"""

import os, sys, time, json, zipfile, re, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ───────────── Config ─────────────
DATA_ROOT  = "/root/autodl-tmp/mind_small"
GLOVE_FILE = "/root/autodl-tmp/mind_small/glove/glove.6B.300d.txt"
SAVE_DIR   = "/root/autodl-tmp/mind_small/baseline_nrms_model"

WORD_EMB_DIM   = 300
NEWS_DIM       = 256
NUM_HEADS      = 16
ATTN_HIDDEN    = 200
TITLE_SIZE     = 30
HIS_SIZE       = 50
NPRATIO        = 4
DROPOUT        = 0.2

EPOCHS         = 1      # 1 epoch for quick verification
BATCH_SIZE     = 64
LR             = 1e-4
SEED           = 42


# ───────────── Tokenizer ─────────────
_SPLIT_RE = re.compile(r"[\w']+|[^\s\w]")

def simple_tokenize(text):
    return _SPLIT_RE.findall(text.lower())


# ───────────── Data Loading ─────────────
def build_word_dict(news_files):
    word_cnt = {}
    for f in news_files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                title = line.strip("\n").split("\t")[3]
                for w in simple_tokenize(title):
                    word_cnt[w] = word_cnt.get(w, 0) + 1
    word_dict = {"<PAD>": 0}
    for w, c in sorted(word_cnt.items(), key=lambda x: -x[1]):
        word_dict[w] = len(word_dict)
    log.info(f"Word dict size: {len(word_dict)}")
    return word_dict


def load_glove(glove_file, word_dict, emb_dim):
    emb = np.random.normal(0, 0.1, (len(word_dict), emb_dim)).astype(np.float32)
    emb[0] = 0  # <PAD>
    found = 0
    with open(glove_file, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in word_dict:
                emb[word_dict[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1
    log.info(f"GloVe: {found}/{len(word_dict)} words found")
    return emb


def title_to_ids(title, word_dict, max_len):
    tokens = simple_tokenize(title)[:max_len]
    ids = [word_dict.get(w, 0) for w in tokens]
    ids += [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)


def load_news(news_file, word_dict, max_len):
    news = {}
    with open(news_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip("\n").split("\t")
            nid, title = parts[0], parts[3]
            news[nid] = title_to_ids(title, word_dict, max_len)
    news["PAD"] = np.zeros(max_len, dtype=np.int64)
    return news


def load_behaviors(beh_file):
    behaviors = []
    with open(beh_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip("\n").split("\t")
            history = parts[3].split() if parts[3] else []
            impressions = parts[4].split() if len(parts) > 4 and parts[4] else []
            behaviors.append({
                "impr_id": parts[0],
                "history": history,
                "impressions": impressions,
            })
    return behaviors


# ───────────── Datasets ─────────────
class TrainDataset(Dataset):
    def __init__(self, news, behaviors, his_size, npratio):
        self.news = news
        self.his_size = his_size
        self.npratio = npratio
        self.samples = []
        for beh in behaviors:
            pos, neg = [], []
            for imp in beh["impressions"]:
                nid, label = imp.rsplit("-", 1)
                (pos if label == "1" else neg).append(nid)
            for p in pos:
                self.samples.append((beh["history"], p, neg))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history, pos_nid, neg_list = self.samples[idx]
        if len(neg_list) >= self.npratio:
            negs = random.sample(neg_list, self.npratio)
        elif neg_list:
            negs = neg_list + random.choices(neg_list, k=self.npratio - len(neg_list))
        else:
            negs = ["PAD"] * self.npratio

        cands = [pos_nid] + negs
        cand_ids = np.stack([self.news.get(n, self.news["PAD"]) for n in cands])

        hist = history[-self.his_size:]
        hist_ids = []
        click_mask = np.zeros(self.his_size, dtype=np.float32)
        for i in range(self.his_size):
            if i < len(hist):
                hist_ids.append(self.news.get(hist[i], self.news["PAD"]))
                click_mask[i] = 1.0
            else:
                hist_ids.append(self.news["PAD"])
        hist_ids = np.stack(hist_ids)

        return {
            "cand_ids": cand_ids,
            "hist_ids": hist_ids,
            "click_mask": click_mask,
            "label": 0,
        }


# ───────────── Model ─────────────
class AdditiveAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, mask=None):
        e = self.query(torch.tanh(self.proj(x))).squeeze(-1)
        if mask is not None:
            bool_mask = mask.bool()
            # If entire row is masked, use uniform attention to avoid NaN
            all_masked = ~bool_mask.any(dim=-1, keepdim=True)
            safe_mask = bool_mask | all_masked
            e = e.masked_fill(~safe_mask, -1e4)
        a = F.softmax(e, dim=-1)
        return torch.bmm(a.unsqueeze(1), x).squeeze(1)


class MHSA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x, mask=None):
        if mask is not None:
            bool_mask = mask.bool()
            # If all positions are masked, don't mask at all (avoid NaN from MHA)
            all_masked = ~bool_mask.any(dim=-1, keepdim=True)
            safe_mask = bool_mask | all_masked
            kpm = ~safe_mask
        else:
            kpm = None
        out, _ = self.attn(x, x, x, key_padding_mask=kpm)
        return out


class NewsEncoder(nn.Module):
    def __init__(self, pretrained_emb, news_dim, num_heads, attn_hidden, dropout):
        super().__init__()
        vocab_size, emb_dim = pretrained_emb.shape
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.embed.weight = nn.Parameter(torch.tensor(pretrained_emb), requires_grad=False)
        self.proj = nn.Linear(emb_dim, news_dim)
        self.mhsa = MHSA(news_dim, num_heads)
        self.attn = AdditiveAttention(news_dim, attn_hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.drop(self.proj(self.embed(x)))
        mask = (x != 0).float()
        y = self.drop(self.mhsa(emb, mask))
        return self.attn(y, mask)


class UserEncoder(nn.Module):
    def __init__(self, news_dim, num_heads, attn_hidden, dropout):
        super().__init__()
        self.mhsa = MHSA(news_dim, num_heads)
        self.attn = AdditiveAttention(news_dim, attn_hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, news_vecs, mask=None):
        y = self.drop(self.mhsa(news_vecs, mask))
        return self.attn(y, mask)


class NRMS(nn.Module):
    def __init__(self, pretrained_emb, news_dim, num_heads, attn_hidden, dropout):
        super().__init__()
        self.news_encoder = NewsEncoder(pretrained_emb, news_dim, num_heads, attn_hidden, dropout)
        self.user_encoder = UserEncoder(news_dim, num_heads, attn_hidden, dropout)
        self.news_dim = news_dim

    def forward(self, cand_ids, hist_ids, click_mask):
        B, C, S = cand_ids.shape
        _, H, _ = hist_ids.shape
        cand_vecs = self.news_encoder(cand_ids.view(B * C, S)).view(B, C, -1)
        hist_vecs = self.news_encoder(hist_ids.view(B * H, S)).view(B, H, -1)
        user_vecs = self.user_encoder(hist_vecs, click_mask)
        return torch.bmm(cand_vecs, user_vecs.unsqueeze(-1)).squeeze(-1)

    def encode_news(self, x):
        return self.news_encoder(x)

    def encode_user(self, news_vecs, mask):
        return self.user_encoder(news_vecs, mask)


# ───────────── Metrics ─────────────
def compute_metrics(labels_list, preds_list):
    aucs, mrrs, n5s, n10s = [], [], [], []
    for labels, preds in zip(labels_list, preds_list):
        labels, preds = np.array(labels), np.array(preds)
        if len(set(labels)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(labels, preds))
        except ValueError:
            pass
        order = np.argsort(-preds)
        rl = labels[order]
        mrr = 0.0
        for i, l in enumerate(rl):
            if l == 1:
                mrr = 1.0 / (i + 1)
                break
        mrrs.append(mrr)
        def dcg(r, k):
            r = r[:k]
            return float(np.sum((2**r - 1) / np.log2(np.arange(2, len(r) + 2))))
        for k, store in [(5, n5s), (10, n10s)]:
            d = dcg(rl, k)
            ideal = dcg(np.sort(labels)[::-1], k)
            store.append(d / ideal if ideal > 0 else 0.0)
    return {
        "group_auc": float(np.mean(aucs)) if aucs else 0,
        "mean_mrr": float(np.mean(mrrs)) if mrrs else 0,
        "ndcg@5": float(np.mean(n5s)) if n5s else 0,
        "ndcg@10": float(np.mean(n10s)) if n10s else 0,
    }


# ───────────── Eval ─────────────
@torch.no_grad()
def evaluate(model, news_dict, eval_behaviors, device, max_eval=None):
    model.eval()
    # Pre-encode all news
    log.info("Pre-encoding news...")
    news_vecs = {}
    nids = [n for n in news_dict if n != "PAD"]
    bs = 1024
    for i in range(0, len(nids), bs):
        batch_nids = nids[i:i+bs]
        inp = torch.tensor(np.stack([news_dict[n] for n in batch_nids])).to(device)
        vecs = model.encode_news(inp).cpu().numpy()
        for j, nid in enumerate(batch_nids):
            news_vecs[nid] = vecs[j]
    pad_vec = np.zeros(model.news_dim, dtype=np.float32)

    log.info("Scoring impressions...")
    all_labels, all_preds, all_ids = [], [], []
    count = 0
    for beh in eval_behaviors:
        impr_news, labels = [], []
        for imp in beh["impressions"]:
            nid, label = imp.rsplit("-", 1)
            impr_news.append(nid)
            labels.append(int(label))
        if not impr_news or len(set(labels)) < 2:
            continue

        hist = beh["history"][-HIS_SIZE:]
        hvecs, cmask = [], []
        for i in range(HIS_SIZE):
            if i < len(hist) and hist[i] in news_vecs:
                hvecs.append(news_vecs[hist[i]])
                cmask.append(1.0)
            else:
                hvecs.append(pad_vec)
                cmask.append(0.0)

        ht = torch.tensor(np.stack(hvecs)).unsqueeze(0).to(device)
        cm = torch.tensor(np.array(cmask, dtype=np.float32)).unsqueeze(0).to(device)
        user_vec = model.encode_user(ht, cm).squeeze(0)

        cvecs = [news_vecs.get(n, pad_vec) for n in impr_news]
        ct = torch.tensor(np.stack(cvecs)).to(device)
        scores = torch.matmul(ct, user_vec).cpu().numpy()

        all_labels.append(labels)
        all_preds.append(scores.tolist())
        all_ids.append(beh["impr_id"])
        count += 1
        if max_eval and count >= max_eval:
            break

    metrics = compute_metrics(all_labels, all_preds)
    return metrics, all_labels, all_preds, all_ids


# ───────────── Main ─────────────
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    train_news_f = os.path.join(DATA_ROOT, "train", "news.tsv")
    valid_news_f = os.path.join(DATA_ROOT, "valid", "news.tsv")
    train_beh_f  = os.path.join(DATA_ROOT, "train", "behaviors.tsv")
    valid_beh_f  = os.path.join(DATA_ROOT, "valid", "behaviors.tsv")

    log.info("Building word dict...")
    word_dict = build_word_dict([train_news_f, valid_news_f])

    log.info("Loading GloVe embeddings...")
    emb_matrix = load_glove(GLOVE_FILE, word_dict, WORD_EMB_DIM)

    log.info("Loading news...")
    train_news = load_news(train_news_f, word_dict, TITLE_SIZE)
    valid_news = load_news(valid_news_f, word_dict, TITLE_SIZE)
    all_news = {**train_news, **valid_news}
    log.info(f"  Total news: {len(all_news) - 1}")

    log.info("Loading behaviors...")
    train_behs = load_behaviors(train_beh_f)
    valid_behs = load_behaviors(valid_beh_f)
    log.info(f"  Train impressions: {len(train_behs)}, Valid: {len(valid_behs)}")

    log.info("Building train dataset...")
    train_ds = TrainDataset(all_news, train_behs, HIS_SIZE, NPRATIO)
    log.info(f"  Train samples: {len(train_ds)}")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)

    log.info("Building NRMS model...")
    model = NRMS(emb_matrix, NEWS_DIM, NUM_HEADS, ATTN_HIDDEN, DROPOUT).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Params: {trainable:,} trainable / {total_params:,} total")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Quick eval before training
    log.info("=== Eval before training (first 1000 impressions) ===")
    m, _, _, _ = evaluate(model, all_news, valid_behs, device, max_eval=1000)
    log.info(f"  {m}")

    # Training
    best_auc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_steps = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            cand = batch["cand_ids"].to(device)
            hist = batch["hist_ids"].to(device)
            cmask = batch["click_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(cand, hist, cmask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

            if (step + 1) % 100 == 0:
                log.info(f"  Epoch {epoch+1} step {step+1}/{len(train_loader)}, "
                         f"avg_loss={total_loss/n_steps:.4f}")

        elapsed = time.time() - t0
        avg = total_loss / max(n_steps, 1)
        log.info(f"Epoch {epoch+1}: avg_loss={avg:.4f}, time={elapsed:.1f}s")

        log.info(f"=== Eval after epoch {epoch+1} ===")
        t0 = time.time()
        metrics, _, _, _ = evaluate(model, all_news, valid_behs, device)
        log.info(f"  {metrics}  (eval time: {time.time()-t0:.1f}s)")

        if metrics["group_auc"] > best_auc:
            best_auc = metrics["group_auc"]
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
            log.info(f"  Saved best model (group_auc={best_auc:.4f})")

    # Generate predictions with best model
    log.info("=== Generating predictions ===")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pt"), weights_only=True))
    metrics, all_labels, all_preds, all_ids = evaluate(model, all_news, valid_behs, device)
    log.info(f"Final metrics: {metrics}")

    pred_path = os.path.join(DATA_ROOT, "prediction_baseline.txt")
    with open(pred_path, "w") as f:
        for impr_id, preds in zip(all_ids, all_preds):
            ranks = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
            f.write(f"{impr_id} [" + ",".join(str(r) for r in ranks) + "]\n")

    zip_path = os.path.join(DATA_ROOT, "prediction_baseline.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction.txt")
    log.info(f"Prediction saved to {zip_path}")

    config = {"news_dim": NEWS_DIM, "epochs": EPOCHS, "lr": LR,
              "best_auc": best_auc, "final_metrics": metrics}
    with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    log.info("All done!")


if __name__ == "__main__":
    main()
