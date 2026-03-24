"""
Training script for BERT-NRMS on MIND dataset.
Uses logging instead of tqdm for clean nohup output.
"""

import os
import sys
import time
import json
import zipfile
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))
from model import BertNRMS
from dataset import load_news, load_behaviors, MINDTrainDataset, MINDEvalDataset


# ──────────────────────────── Config ────────────────────────────

DATA_PATH = "/root/autodl-tmp/mind_small"
MODEL_SAVE_PATH = "/root/autodl-tmp/mind_small/bert_nrms_model"
BERT_MODEL = "/root/autodl-tmp/bert-base-uncased"
HF_MIRROR = "https://hf-mirror.com"

EPOCHS = 5
BATCH_SIZE = 32
LR = 2e-5
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
NEWS_DIM = 256
NUM_HEADS = 16
ATTENTION_HIDDEN = 200
HIS_SIZE = 50
NPRATIO = 4
MAX_TITLE_LEN = 30
FREEZE_BERT_LAYERS = 8
GRAD_ACCUM_STEPS = 2
SEED = 42


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def compute_metrics(labels_list, preds_list):
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    for labels, preds in zip(labels_list, preds_list):
        labels = np.array(labels)
        preds = np.array(preds)
        if len(set(labels)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(labels, preds))
        except ValueError:
            pass
        order = np.argsort(-preds)
        ranked_labels = labels[order]
        mrr = 0.0
        for i, l in enumerate(ranked_labels):
            if l == 1:
                mrr = 1.0 / (i + 1)
                break
        mrrs.append(mrr)
        def dcg(rel, k):
            rel = rel[:k]
            return np.sum((2**rel - 1) / np.log2(np.arange(2, len(rel) + 2)))
        for k, store in [(5, ndcg5s), (10, ndcg10s)]:
            d = dcg(ranked_labels, k)
            ideal = dcg(np.sort(labels)[::-1], k)
            store.append(d / ideal if ideal > 0 else 0.0)
    return {
        "group_auc": float(np.mean(aucs)) if aucs else 0,
        "mean_mrr": float(np.mean(mrrs)) if mrrs else 0,
        "ndcg@5": float(np.mean(ndcg5s)) if ndcg5s else 0,
        "ndcg@10": float(np.mean(ndcg10s)) if ndcg10s else 0,
    }


@torch.no_grad()
def evaluate(model, news_dict, eval_dataset, device, max_eval=None):
    model.eval()
    log.info("Pre-encoding news...")
    news_vecs = {}
    all_nids = [nid for nid in news_dict if nid != "PAD"]
    batch_size = 512
    for i in range(0, len(all_nids), batch_size):
        batch_nids = all_nids[i:i + batch_size]
        input_ids = torch.tensor(
            np.stack([news_dict[n]["input_ids"] for n in batch_nids])
        ).to(device)
        attn_mask = torch.tensor(
            np.stack([news_dict[n]["attention_mask"] for n in batch_nids])
        ).to(device)
        with autocast("cuda"):
            vecs = model.encode_news(input_ids, attn_mask)
        vecs = vecs.float().cpu().numpy()
        for j, nid in enumerate(batch_nids):
            news_vecs[nid] = vecs[j]
        if (i // batch_size) % 20 == 0:
            log.info(f"  News encoding: {i}/{len(all_nids)}")

    pad_vec = np.zeros(model.news_dim, dtype=np.float32)
    log.info("Scoring impressions...")
    all_labels, all_preds, all_impr_ids = [], [], []
    count = 0
    for sample in eval_dataset.samples:
        history, impr_news, labels, impr_id = sample
        hist = history[-HIS_SIZE:]
        hist_vecs, clicked_mask_list = [], []
        for i in range(HIS_SIZE):
            if i < len(hist) and hist[i] in news_vecs:
                hist_vecs.append(news_vecs[hist[i]])
                clicked_mask_list.append(1.0)
            else:
                hist_vecs.append(pad_vec)
                clicked_mask_list.append(0.0)

        hist_tensor = torch.tensor(np.stack(hist_vecs)).unsqueeze(0).to(device)
        clicked_mask = torch.tensor(np.array(clicked_mask_list)).unsqueeze(0).to(device)
        with autocast("cuda"):
            user_vec = model.encode_user(hist_tensor, clicked_mask)

        cand_vecs = [news_vecs.get(nid, pad_vec) for nid in impr_news]
        cand_tensor = torch.tensor(np.stack(cand_vecs)).to(device).float()
        scores = torch.matmul(cand_tensor, user_vec.float().squeeze(0)).cpu().numpy()

        all_labels.append(labels)
        all_preds.append(scores.tolist())
        all_impr_ids.append(impr_id)
        count += 1
        if count % 10000 == 0:
            log.info(f"  Scored {count} impressions...")
        if max_eval and count >= max_eval:
            break

    metrics = compute_metrics(all_labels, all_preds)
    return metrics, all_labels, all_preds, all_impr_ids


def train():
    set_seed(SEED)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    os.environ["HF_ENDPOINT"] = HF_MIRROR
    log.info(f"Loading tokenizer: {BERT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    train_news_file = os.path.join(DATA_PATH, "train", "news.tsv")
    train_behaviors_file = os.path.join(DATA_PATH, "train", "behaviors.tsv")
    valid_news_file = os.path.join(DATA_PATH, "valid", "news.tsv")
    valid_behaviors_file = os.path.join(DATA_PATH, "valid", "behaviors.tsv")

    log.info("Loading and tokenizing news...")
    train_news = load_news(train_news_file, tokenizer, MAX_TITLE_LEN)
    valid_news = load_news(valid_news_file, tokenizer, MAX_TITLE_LEN)
    all_news = {**train_news, **valid_news}
    log.info(f"  Total news articles: {len(all_news) - 1}")

    log.info("Loading behaviors...")
    train_behaviors = load_behaviors(train_behaviors_file)
    valid_behaviors = load_behaviors(valid_behaviors_file)
    log.info(f"  Train impressions: {len(train_behaviors)}")
    log.info(f"  Valid impressions: {len(valid_behaviors)}")

    log.info("Building datasets...")
    train_dataset = MINDTrainDataset(all_news, train_behaviors, HIS_SIZE, NPRATIO, MAX_TITLE_LEN)
    eval_dataset = MINDEvalDataset(all_news, valid_behaviors, HIS_SIZE, MAX_TITLE_LEN)
    log.info(f"  Train samples: {len(train_dataset)}")
    log.info(f"  Eval impressions: {len(eval_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    log.info(f"Building BERT-NRMS model (freeze first {FREEZE_BERT_LAYERS} BERT layers)...")
    model = BertNRMS(
        bert_model_name=BERT_MODEL,
        news_dim=NEWS_DIM,
        num_heads=NUM_HEADS,
        attention_hidden_dim=ATTENTION_HIDDEN,
        freeze_bert_layers=FREEZE_BERT_LAYERS,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Trainable params: {trainable:,} / {total_params:,} ({100*trainable/total_params:.1f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()
    log.info(f"  Total optim steps: {total_steps}, warmup: {warmup_steps}")

    # Quick eval before training
    log.info("=== Eval before training (first 2000 impressions) ===")
    metrics, _, _, _ = evaluate(model, all_news, eval_dataset, device, max_eval=2000)
    log.info(f"  {metrics}")

    # Training loop
    best_auc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        num_valid_steps = 0
        step = 0
        optimizer.zero_grad()
        t0 = time.time()
        n_batches = len(train_loader)

        for batch in train_loader:
            cand_ids = batch["candidate_input_ids"].to(device)
            cand_mask = batch["candidate_attention_mask"].to(device)
            click_ids = batch["clicked_input_ids"].to(device)
            click_mask = batch["clicked_attention_mask"].to(device)
            clicked_mask = batch["clicked_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast("cuda"):
                logits = model(cand_ids, cand_mask, click_ids, click_mask, clicked_mask)
                loss = criterion(logits, labels)
                loss = loss / GRAD_ACCUM_STEPS

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                step += 1
                continue

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    MAX_GRAD_NORM,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            num_valid_steps += 1
            step += 1

            if step % 200 == 0:
                avg_loss = total_loss / max(num_valid_steps, 1)
                cur_lr = scheduler.get_last_lr()[0]
                log.info(f"  Epoch {epoch+1} step {step}/{n_batches}, "
                         f"avg_loss={avg_loss:.4f}, lr={cur_lr:.2e}")

        train_time = time.time() - t0
        avg_loss = total_loss / max(num_valid_steps, 1)
        log.info(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}, time={train_time:.1f}s")

        log.info(f"=== Eval after epoch {epoch+1} ===")
        t0 = time.time()
        metrics, _, _, _ = evaluate(model, all_news, eval_dataset, device)
        eval_time = time.time() - t0
        log.info(f"  {metrics}  (eval time: {eval_time:.1f}s)")

        if metrics["group_auc"] > best_auc:
            best_auc = metrics["group_auc"]
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "best_model.pt"))
            log.info(f"  Saved best model (group_auc={best_auc:.4f})")

    # Load best and generate predictions
    log.info("=== Generating predictions with best model ===")
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, "best_model.pt"), weights_only=True))
    metrics, all_labels, all_preds, all_impr_ids = evaluate(model, all_news, eval_dataset, device)
    log.info(f"Final metrics: {metrics}")

    pred_path = os.path.join(DATA_PATH, "prediction_bert.txt")
    with open(pred_path, "w") as f:
        for impr_id, preds in zip(all_impr_ids, all_preds):
            pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
            pred_rank_str = "[" + ",".join(str(r) for r in pred_rank) + "]"
            f.write(f"{impr_id} {pred_rank_str}\n")

    zip_path = os.path.join(DATA_PATH, "prediction_bert.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction.txt")
    log.info(f"Predictions saved to {zip_path}")

    config = {
        "bert_model": BERT_MODEL, "news_dim": NEWS_DIM, "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "lr": LR, "freeze_layers": FREEZE_BERT_LAYERS,
        "best_group_auc": best_auc, "final_metrics": metrics,
    }
    with open(os.path.join(MODEL_SAVE_PATH, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    log.info("All done!")


if __name__ == "__main__":
    train()
