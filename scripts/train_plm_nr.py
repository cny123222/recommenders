"""
PLM-NR (Pre-trained Language Model based News Recommendation) with RoBERTa.

Key differences vs our BERT-NRMS:
  1. RoBERTa instead of BERT (no token_type_ids)
  2. Truncated to 8 transformer layers (not 12), only fine-tune last 2 layers
  3. News Encoder: RoBERTa → Multi-Head Self-Attention → Additive Attention (extra MHSA layer)
  4. User Encoder: simplified — only Additive Attention (no Self-Attention)
  5. news_dim = 64, num_attn_heads = 20, head_dim = 20

Based on: https://github.com/wuch15/PLM4NewsRec

Usage:
  nohup python -u scripts/train_plm_nr.py > /root/autodl-tmp/plm_nr_large.log 2>&1 &
"""

import os, sys, time, json, zipfile, re, logging, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ───────────── Config ─────────────
DATA_ROOT        = "/root/autodl-tmp/mind_large"
ROBERTA_MODEL    = "/root/autodl-tmp/roberta-base"
SAVE_DIR         = "/root/autodl-tmp/mind_large/plm_nr_model"

EPOCHS           = 3
BATCH_SIZE       = 64
LR               = 1e-4
WARMUP_RATIO     = 0.1
MAX_GRAD_NORM    = 1.0
NEWS_DIM         = 64
NUM_ATTN_HEADS   = 20
HEAD_DIM         = 20
QUERY_DIM        = 200
MAX_TITLE_LEN    = 24
HIS_SIZE         = 50
NPRATIO          = 4
NUM_BERT_LAYERS  = 8
FINETUNE_LAYERS  = 2      # fine-tune last N of the 8 layers
GRAD_ACCUM       = 2
SEED             = 42
EVAL_SAMPLE      = 5000
DROP_RATE        = 0.2

# ───────────── Attention Modules (PLM-NR style) ─────────────

class AdditiveAttention(nn.Module):
    """exp * mask / (sum + eps) style — naturally handles all-masked case."""
    def __init__(self, d_h, hidden_size=200):
        super().__init__()
        self.fc1 = nn.Linear(d_h, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, mask=None):
        e = self.fc2(torch.tanh(self.fc1(x)))          # (B, L, 1)
        alpha = torch.exp(e)
        if mask is not None:
            alpha = alpha * mask.unsqueeze(2).float()
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
        return torch.bmm(x.permute(0, 2, 1), alpha).squeeze(-1)  # (B, D)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        scores = torch.exp(scores)
        if mask is not None:
            scores = scores * mask
        attn = scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)
        return torch.matmul(attn, V)


class MultiHeadAttention(nn.Module):
    """Custom MHA from PLM-NR — uses ScaledDotProduct with exp*mask normalization."""
    def __init__(self, d_model, n_heads, d_k, d_v):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.sdpa = ScaledDotProductAttention(d_k)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        q = self.W_Q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(x).view(B, L, self.n_heads, self.d_v).transpose(1, 2)

        if mask is not None:
            # (B, L) → (B, L, L) → (B, n_heads, L, L)
            mask_2d = mask.unsqueeze(1).expand(B, L, L)
            mask_2d = mask_2d.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        else:
            mask_2d = None

        ctx = self.sdpa(q, k, v, mask_2d)                 # (B, H, L, d_v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, self.n_heads * self.d_v)
        return ctx


# ───────────── News Encoder ─────────────
class PLMNewsEncoder(nn.Module):
    """
    RoBERTa (8 layers) → dropout → MultiHead Self-Attention → dropout →
    Additive Attention → reduce_dim_linear → news_vec
    """
    def __init__(self, roberta_path, news_dim, n_heads, d_k, d_v, query_dim,
                 num_layers, finetune_layers, drop_rate):
        super().__init__()
        config = AutoConfig.from_pretrained(roberta_path, output_hidden_states=True)
        config.num_hidden_layers = num_layers
        self.roberta = AutoModel.from_pretrained(roberta_path, config=config)
        hidden = config.hidden_size  # 768

        freeze_below = num_layers - finetune_layers
        for name, param in self.roberta.named_parameters():
            if "embeddings" in name:
                param.requires_grad = False
            elif "encoder.layer." in name:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_num < freeze_below:
                    param.requires_grad = False

        mha_out_dim = n_heads * d_v
        self.mha = MultiHeadAttention(hidden, n_heads, d_k, d_v)
        self.additive_attn = AdditiveAttention(mha_out_dim, query_dim)
        self.reduce_dim = nn.Linear(mha_out_dim, news_dim)
        self.drop_rate = drop_rate

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # out.hidden_states: tuple of (num_layers+1) tensors; last one = layer output
        hidden = out.hidden_states[-1]                    # (B, L, 768)
        hidden = F.dropout(hidden, p=self.drop_rate, training=self.training)

        mask = attention_mask.float()
        mha_out = self.mha(hidden, mask)                  # (B, L, n_heads*d_v)
        mha_out = F.dropout(mha_out, p=self.drop_rate, training=self.training)

        news_vec = self.additive_attn(mha_out, mask)      # (B, n_heads*d_v)
        news_vec = self.reduce_dim(news_vec)               # (B, news_dim)
        return news_vec


# ───────────── User Encoder (simplified, PLM-NR style) ─────────────
class PLMUserEncoder(nn.Module):
    """Only Additive Attention on clicked news vectors — no Self-Attention."""
    def __init__(self, news_dim, query_dim):
        super().__init__()
        self.attn = AdditiveAttention(news_dim, query_dim)

    def forward(self, news_vecs, mask=None):
        return self.attn(news_vecs, mask)                  # (B, news_dim)


# ───────────── Full Model ─────────────
class PLMNR(nn.Module):
    def __init__(self, roberta_path, news_dim, n_heads, d_k, d_v, query_dim,
                 num_layers, finetune_layers, drop_rate):
        super().__init__()
        self.news_encoder = PLMNewsEncoder(
            roberta_path, news_dim, n_heads, d_k, d_v, query_dim,
            num_layers, finetune_layers, drop_rate)
        self.user_encoder = PLMUserEncoder(news_dim, query_dim)
        self.news_dim = news_dim

    def encode_news(self, input_ids, attention_mask):
        return self.news_encoder(input_ids, attention_mask)

    def encode_user(self, news_vecs, mask):
        return self.user_encoder(news_vecs, mask)

    def forward(self, cand_ids, cand_mask, click_ids, click_mask, clicked_mask):
        B, C, S = cand_ids.shape
        H = click_ids.size(1)
        cv = self.encode_news(cand_ids.view(-1, S), cand_mask.view(-1, S)).view(B, C, -1)
        hv = self.encode_news(click_ids.view(-1, S), click_mask.view(-1, S)).view(B, H, -1)
        uv = self.encode_user(hv, clicked_mask)
        return torch.bmm(cv, uv.unsqueeze(-1)).squeeze(-1)


# ───────────── Data ─────────────
def load_news(news_file, tokenizer, max_len):
    news = {}
    with open(news_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip("\n").split("\t")
            nid, title = parts[0], parts[3]
            enc = tokenizer(title, max_length=max_len, padding="max_length",
                            truncation=True, return_tensors="np")
            news[nid] = {
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
            }
    news["PAD"] = {
        "input_ids": np.zeros(max_len, dtype=np.int64),
        "attention_mask": np.zeros(max_len, dtype=np.int64),
    }
    return news


def load_behaviors_train(beh_file):
    behaviors = []
    with open(beh_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip("\n").split("\t")
            history = parts[3].split() if parts[3] else []
            impressions = parts[4].split() if len(parts) > 4 and parts[4] else []
            behaviors.append({"impr_id": parts[0], "history": history, "impressions": impressions})
    return behaviors


def load_behaviors_test(beh_file):
    behaviors = []
    with open(beh_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip("\n").split("\t")
            history = parts[3].split() if parts[3] else []
            impr_news = parts[4].split() if len(parts) > 4 and parts[4] else []
            behaviors.append({"impr_id": parts[0], "history": history, "impr_news": impr_news})
    return behaviors


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

    def _feat(self, nid):
        e = self.news.get(nid, self.news["PAD"])
        return e["input_ids"], e["attention_mask"]

    def __getitem__(self, idx):
        history, pos_nid, neg_list = self.samples[idx]
        if len(neg_list) >= self.npratio:
            negs = random.sample(neg_list, self.npratio)
        elif neg_list:
            negs = neg_list + random.choices(neg_list, k=self.npratio - len(neg_list))
        else:
            negs = ["PAD"] * self.npratio

        cands = [pos_nid] + negs
        cand_ids = np.stack([self._feat(n)[0] for n in cands])
        cand_mask = np.stack([self._feat(n)[1] for n in cands])

        hist = history[-self.his_size:]
        h_ids, h_mask, c_mask = [], [], np.zeros(self.his_size, dtype=np.float32)
        for i in range(self.his_size):
            if i < len(hist):
                ids, mask = self._feat(hist[i])
                h_ids.append(ids); h_mask.append(mask); c_mask[i] = 1.0
            else:
                h_ids.append(self.news["PAD"]["input_ids"])
                h_mask.append(self.news["PAD"]["attention_mask"])

        return {
            "candidate_input_ids": cand_ids.astype(np.int64),
            "candidate_attention_mask": cand_mask.astype(np.int64),
            "clicked_input_ids": np.stack(h_ids).astype(np.int64),
            "clicked_attention_mask": np.stack(h_mask).astype(np.int64),
            "clicked_mask": c_mask,
            "label": 0,
        }


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
                mrr = 1.0 / (i + 1); break
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


# ───────────── Eval / Predict ─────────────
@torch.no_grad()
def encode_all_news(model, news_dict, device):
    news_vecs = {}
    nids = [n for n in news_dict if n != "PAD"]
    bs = 512
    for i in range(0, len(nids), bs):
        batch_nids = nids[i:i+bs]
        inp = torch.tensor(np.stack([news_dict[n]["input_ids"] for n in batch_nids])).to(device)
        mask = torch.tensor(np.stack([news_dict[n]["attention_mask"] for n in batch_nids])).to(device)
        with autocast("cuda"):
            vecs = model.encode_news(inp, mask)
        vecs = vecs.float().cpu().numpy()
        for j, nid in enumerate(batch_nids):
            news_vecs[nid] = vecs[j]
        if (i // bs) % 50 == 0:
            log.info(f"  News encoding: {i}/{len(nids)}")
    return news_vecs


@torch.no_grad()
def score_user(model, news_vecs, history, device, pad_vec):
    hist = history[-HIS_SIZE:]
    hvecs, cmask = [], []
    for i in range(HIS_SIZE):
        if i < len(hist) and hist[i] in news_vecs:
            hvecs.append(news_vecs[hist[i]]); cmask.append(1.0)
        else:
            hvecs.append(pad_vec); cmask.append(0.0)
    ht = torch.tensor(np.stack(hvecs)).unsqueeze(0).to(device)
    cm = torch.tensor(np.array(cmask, dtype=np.float32)).unsqueeze(0).to(device)
    with autocast("cuda"):
        uv = model.encode_user(ht, cm)
    return uv.float().squeeze(0)


@torch.no_grad()
def evaluate_dev(model, news_vecs, dev_behaviors, device, max_eval=None):
    model.eval()
    pad_vec = np.zeros(model.news_dim, dtype=np.float32)
    all_labels, all_preds = [], []
    count = 0
    for beh in dev_behaviors:
        impr_news, labels = [], []
        for imp in beh["impressions"]:
            nid, label = imp.rsplit("-", 1)
            impr_news.append(nid); labels.append(int(label))
        if not impr_news or len(set(labels)) < 2:
            continue
        uv = score_user(model, news_vecs, beh["history"], device, pad_vec)
        cvecs = [news_vecs.get(n, pad_vec) for n in impr_news]
        ct = torch.tensor(np.stack(cvecs)).to(device).float()
        scores = torch.matmul(ct, uv).cpu().numpy()
        all_labels.append(labels); all_preds.append(scores.tolist())
        count += 1
        if count % 10000 == 0:
            log.info(f"  Scored {count} impressions...")
        if max_eval and count >= max_eval:
            break
    return compute_metrics(all_labels, all_preds)


@torch.no_grad()
def predict_test(model, news_vecs, test_behaviors, device):
    model.eval()
    pad_vec = np.zeros(model.news_dim, dtype=np.float32)
    all_ids, all_preds = [], []
    for i, beh in enumerate(test_behaviors):
        impr_news = beh["impr_news"]
        if not impr_news:
            all_ids.append(beh["impr_id"]); all_preds.append([]); continue
        uv = score_user(model, news_vecs, beh["history"], device, pad_vec)
        cvecs = [news_vecs.get(n, pad_vec) for n in impr_news]
        ct = torch.tensor(np.stack(cvecs)).to(device).float()
        scores = torch.matmul(ct, uv).cpu().numpy()
        all_ids.append(beh["impr_id"]); all_preds.append(scores.tolist())
        if (i + 1) % 50000 == 0:
            log.info(f"  Test prediction: {i+1}/{len(test_behaviors)}")
    return all_ids, all_preds


# ───────────── Main ─────────────
def main():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    log.info(f"Loading tokenizer: {ROBERTA_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)

    log.info("Loading and tokenizing news (train + dev + test)...")
    train_news = load_news(os.path.join(DATA_ROOT, "train", "news.tsv"), tokenizer, MAX_TITLE_LEN)
    dev_news = load_news(os.path.join(DATA_ROOT, "dev", "news.tsv"), tokenizer, MAX_TITLE_LEN)
    test_news = load_news(os.path.join(DATA_ROOT, "test", "news.tsv"), tokenizer, MAX_TITLE_LEN)
    all_news = {**train_news, **dev_news, **test_news}
    log.info(f"  Total news articles: {len(all_news) - 1}")

    log.info("Loading behaviors...")
    train_behs = load_behaviors_train(os.path.join(DATA_ROOT, "train", "behaviors.tsv"))
    dev_behs = load_behaviors_train(os.path.join(DATA_ROOT, "dev", "behaviors.tsv"))
    test_behs = load_behaviors_test(os.path.join(DATA_ROOT, "test", "behaviors.tsv"))
    log.info(f"  Train: {len(train_behs)}, Dev: {len(dev_behs)}, Test: {len(test_behs)}")

    log.info("Building train dataset...")
    train_ds = TrainDataset(all_news, train_behs, HIS_SIZE, NPRATIO)
    log.info(f"  Train samples: {len(train_ds)}")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)

    log.info("Building PLM-NR (RoBERTa) model...")
    model = PLMNR(
        ROBERTA_MODEL, NEWS_DIM, NUM_ATTN_HEADS, HEAD_DIM, HEAD_DIM,
        QUERY_DIM, NUM_BERT_LAYERS, FINETUNE_LAYERS, DROP_RATE
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    log.info(f"  Params: {trainable:,} trainable / {total_p:,} total ({100*trainable/total_p:.1f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()
    log.info(f"  Total optim steps: {total_steps}, warmup: {warmup_steps}")

    best_auc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0; n_valid = 0; step = 0
        optimizer.zero_grad()
        t0 = time.time()
        n_batches = len(train_loader)

        for batch in train_loader:
            ci = batch["candidate_input_ids"].to(device)
            cm = batch["candidate_attention_mask"].to(device)
            hi = batch["clicked_input_ids"].to(device)
            hm = batch["clicked_attention_mask"].to(device)
            ck = batch["clicked_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast("cuda"):
                logits = model(ci, cm, hi, hm, ck)
                loss = criterion(logits, labels) / GRAD_ACCUM

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(); step += 1; continue

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM)
                scaler.step(optimizer); scaler.update()
                scheduler.step(); optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM
            n_valid += 1; step += 1

            if step % 500 == 0:
                avg = total_loss / max(n_valid, 1)
                lr = scheduler.get_last_lr()[0]
                log.info(f"  Epoch {epoch+1} step {step}/{n_batches}, avg_loss={avg:.4f}, lr={lr:.2e}")

        elapsed = time.time() - t0
        avg = total_loss / max(n_valid, 1)
        log.info(f"Epoch {epoch+1}: avg_loss={avg:.4f}, time={elapsed:.1f}s")

        log.info(f"=== Eval after epoch {epoch+1} (first {EVAL_SAMPLE} dev impressions) ===")
        t0 = time.time()
        log.info("Pre-encoding news for eval...")
        news_vecs = encode_all_news(model, all_news, device)
        metrics = evaluate_dev(model, news_vecs, dev_behs, device, max_eval=EVAL_SAMPLE)
        log.info(f"  {metrics}  (eval time: {time.time()-t0:.1f}s)")

        if metrics["group_auc"] > best_auc:
            best_auc = metrics["group_auc"]
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
            log.info(f"  Saved best model (group_auc={best_auc:.4f})")

    # Final evaluation and test prediction
    log.info("=== Loading best model for test prediction ===")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pt"), weights_only=True))
    model.eval()

    log.info("Pre-encoding all news for test...")
    news_vecs = encode_all_news(model, all_news, device)

    log.info("Full dev evaluation...")
    t0 = time.time()
    dev_metrics = evaluate_dev(model, news_vecs, dev_behs, device)
    log.info(f"  Dev metrics: {dev_metrics}  (time: {time.time()-t0:.1f}s)")

    log.info("Generating test predictions...")
    t0 = time.time()
    test_ids, test_preds = predict_test(model, news_vecs, test_behs, device)
    log.info(f"  Test predictions done ({len(test_ids)} impressions, time: {time.time()-t0:.1f}s)")

    pred_path = os.path.join(SAVE_DIR, "prediction.txt")
    with open(pred_path, "w") as f:
        for impr_id, preds in zip(test_ids, test_preds):
            if preds:
                ranks = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
                f.write(f"{impr_id} [" + ",".join(str(r) for r in ranks) + "]\n")
            else:
                f.write(f"{impr_id} []\n")

    zip_path = os.path.join(SAVE_DIR, "prediction.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction.txt")
    log.info(f"Prediction saved to {zip_path}")

    config = {
        "model": "PLM-NR (RoBERTa)",
        "roberta": ROBERTA_MODEL,
        "news_dim": NEWS_DIM,
        "num_attn_heads": NUM_ATTN_HEADS,
        "head_dim": HEAD_DIM,
        "num_bert_layers": NUM_BERT_LAYERS,
        "finetune_layers": FINETUNE_LAYERS,
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "npratio": NPRATIO,
        "best_dev_auc": best_auc,
        "final_dev_metrics": dev_metrics,
    }
    with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    log.info("All done!")


if __name__ == "__main__":
    main()
