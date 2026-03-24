"""
BERT-NRMS training on MIND-large + test set prediction generation.
Uses logging (no tqdm) for clean nohup output.

Usage:
  nohup python -u scripts/train_large.py > /root/autodl-tmp/bert_large.log 2>&1 &
"""

import os, sys, time, json, zipfile, re, logging, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ───────────── Config ─────────────
DATA_ROOT     = "/root/autodl-tmp/mind_large"
GLOVE_FILE    = "/root/autodl-tmp/mind_small/glove/glove.6B.300d.txt"
BERT_MODEL    = "/root/autodl-tmp/bert-base-uncased"
SAVE_DIR      = "/root/autodl-tmp/mind_large/bert_nrms_model"

EPOCHS           = 3
BATCH_SIZE       = 32
LR               = 2e-5
WARMUP_RATIO     = 0.1
MAX_GRAD_NORM    = 1.0
NEWS_DIM         = 256
NUM_HEADS        = 16
ATTN_HIDDEN      = 200
MAX_TITLE_LEN    = 30
HIS_SIZE         = 50
NPRATIO          = 4
FREEZE_BERT      = 8
GRAD_ACCUM       = 2
SEED             = 42
EVAL_SAMPLE      = 5000   # subsample dev for quick eval during training


# ───────────── Model ─────────────
class AdditiveAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, mask=None):
        e = self.query(torch.tanh(self.proj(x))).squeeze(-1)
        if mask is not None:
            bm = mask.bool()
            all_masked = ~bm.any(dim=-1, keepdim=True)
            e = e.masked_fill(~(bm | all_masked), -1e4)
        return torch.bmm(F.softmax(e, dim=-1).unsqueeze(1), x).squeeze(1)


class MHSA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x, mask=None):
        if mask is not None:
            bm = mask.bool()
            all_masked = ~bm.any(dim=-1, keepdim=True)
            kpm = ~(bm | all_masked)
        else:
            kpm = None
        out, _ = self.attn(x, x, x, key_padding_mask=kpm)
        return out


class BertNewsEncoder(nn.Module):
    def __init__(self, bert_model_name, news_dim, freeze_layers=8):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size
        if freeze_layers > 0:
            for name, param in self.bert.named_parameters():
                if "embeddings" in name:
                    param.requires_grad = False
                elif "encoder.layer." in name:
                    layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                    if layer_num < freeze_layers:
                        param.requires_grad = False
        self.attention = AdditiveAttention(bert_hidden, news_dim)
        self.output_proj = nn.Linear(bert_hidden, news_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(out.last_hidden_state)
        v = self.attention(h, mask=attention_mask.bool())
        return self.output_proj(v)


class UserEncoder(nn.Module):
    def __init__(self, news_dim, num_heads, attn_hidden):
        super().__init__()
        self.mhsa = MHSA(news_dim, num_heads)
        self.attn = AdditiveAttention(news_dim, attn_hidden)
        self.drop = nn.Dropout(0.2)

    def forward(self, news_vecs, mask=None):
        y = self.drop(self.mhsa(news_vecs, mask))
        return self.attn(y, mask)


class BertNRMS(nn.Module):
    def __init__(self, bert_model_name, news_dim, num_heads, attn_hidden, freeze_layers):
        super().__init__()
        self.news_encoder = BertNewsEncoder(bert_model_name, news_dim, freeze_layers)
        self.user_encoder = UserEncoder(news_dim, num_heads, attn_hidden)
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
def load_news_bert(news_file, tokenizer, max_len):
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
            behaviors.append({
                "impr_id": parts[0], "history": history, "impressions": impressions,
            })
    return behaviors


def load_behaviors_test(beh_file):
    """Load test behaviors — no labels in impressions."""
    behaviors = []
    with open(beh_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip("\n").split("\t")
            history = parts[3].split() if parts[3] else []
            # Test impressions: just news IDs, no labels
            impr_news = parts[4].split() if len(parts) > 4 and parts[4] else []
            behaviors.append({
                "impr_id": parts[0], "history": history, "impr_news": impr_news,
            })
    return behaviors


class TrainDataset(Dataset):
    def __init__(self, news, behaviors, his_size, npratio, max_title_len):
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
    """Pre-encode all news articles into vectors."""
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
    all_labels, all_preds, all_ids = [], [], []
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
        all_labels.append(labels); all_preds.append(scores.tolist()); all_ids.append(beh["impr_id"])
        count += 1
        if count % 10000 == 0:
            log.info(f"  Scored {count} impressions...")
        if max_eval and count >= max_eval:
            break
    return compute_metrics(all_labels, all_preds)


@torch.no_grad()
def predict_test(model, news_vecs, test_behaviors, device):
    """Generate predictions for test set (no labels)."""
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
    log.info(f"Loading tokenizer: {BERT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    # Load news from all splits
    log.info("Loading and tokenizing news (train + dev + test)...")
    train_news = load_news_bert(os.path.join(DATA_ROOT, "train", "news.tsv"), tokenizer, MAX_TITLE_LEN)
    dev_news = load_news_bert(os.path.join(DATA_ROOT, "dev", "news.tsv"), tokenizer, MAX_TITLE_LEN)
    test_news = load_news_bert(os.path.join(DATA_ROOT, "test", "news.tsv"), tokenizer, MAX_TITLE_LEN)
    all_news = {**train_news, **dev_news, **test_news}
    log.info(f"  Total news articles: {len(all_news) - 1}")

    log.info("Loading behaviors...")
    train_behs = load_behaviors_train(os.path.join(DATA_ROOT, "train", "behaviors.tsv"))
    dev_behs = load_behaviors_train(os.path.join(DATA_ROOT, "dev", "behaviors.tsv"))
    test_behs = load_behaviors_test(os.path.join(DATA_ROOT, "test", "behaviors.tsv"))
    log.info(f"  Train: {len(train_behs)}, Dev: {len(dev_behs)}, Test: {len(test_behs)}")

    log.info("Building train dataset...")
    train_ds = TrainDataset(all_news, train_behs, HIS_SIZE, NPRATIO, MAX_TITLE_LEN)
    log.info(f"  Train samples: {len(train_ds)}")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)

    log.info("Building BERT-NRMS model...")
    model = BertNRMS(BERT_MODEL, NEWS_DIM, NUM_HEADS, ATTN_HIDDEN, FREEZE_BERT).to(device)
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

    # Training loop
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

        # Eval on dev subset
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

    # Generate test predictions with best model
    log.info("=== Loading best model for test prediction ===")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pt"), weights_only=True))
    model.eval()

    log.info("Pre-encoding all news for test...")
    news_vecs = encode_all_news(model, all_news, device)

    # Full dev eval
    log.info("Full dev evaluation...")
    t0 = time.time()
    dev_metrics = evaluate_dev(model, news_vecs, dev_behs, device)
    log.info(f"  Dev metrics: {dev_metrics}  (time: {time.time()-t0:.1f}s)")

    # Test predictions
    log.info("Generating test predictions...")
    t0 = time.time()
    test_ids, test_preds = predict_test(model, news_vecs, test_behs, device)
    log.info(f"  Test predictions done ({len(test_ids)} impressions, time: {time.time()-t0:.1f}s)")

    # Write prediction file
    pred_path = os.path.join(DATA_ROOT, "prediction.txt")
    with open(pred_path, "w") as f:
        for impr_id, preds in zip(test_ids, test_preds):
            if preds:
                ranks = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
                f.write(f"{impr_id} [" + ",".join(str(r) for r in ranks) + "]\n")
            else:
                f.write(f"{impr_id} []\n")

    zip_path = os.path.join(DATA_ROOT, "prediction.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction.txt")
    log.info(f"Prediction saved to {zip_path}")

    config = {
        "model": "BERT-NRMS", "bert": BERT_MODEL, "news_dim": NEWS_DIM,
        "epochs": EPOCHS, "lr": LR, "freeze_layers": FREEZE_BERT,
        "best_dev_auc": best_auc, "final_dev_metrics": dev_metrics,
    }
    with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    log.info("All done!")


if __name__ == "__main__":
    main()
