"""
MIND dataset loader for BERT-NRMS.
Tokenizes news titles with BERT tokenizer, constructs training/eval samples.
"""

import os
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def load_news(news_file, tokenizer, max_title_len=30):
    """Load news and tokenize titles with BERT tokenizer.

    Returns:
        dict: news_id -> {input_ids, attention_mask}
    """
    news = {}
    with open(news_file, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.strip("\n").split("\t")
            nid = fields[0]
            title = fields[3]
            encoded = tokenizer(
                title,
                max_length=max_title_len,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            news[nid] = {
                "input_ids": encoded["input_ids"][0],
                "attention_mask": encoded["attention_mask"][0],
            }
    # Add a padding news entry for empty history slots
    news["PAD"] = {
        "input_ids": np.zeros(max_title_len, dtype=np.int64),
        "attention_mask": np.zeros(max_title_len, dtype=np.int64),
    }
    return news


def load_behaviors(behaviors_file):
    """Load behavior logs.

    Returns:
        list of dicts with keys: impr_id, user, time, history, impressions
    """
    behaviors = []
    with open(behaviors_file, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.strip("\n").split("\t")
            impr_id = fields[0]
            user = fields[1]
            time = fields[2]
            history = fields[3].split() if fields[3] else []
            impressions = fields[4].split() if len(fields) > 4 and fields[4] else []
            behaviors.append({
                "impr_id": impr_id,
                "user": user,
                "time": time,
                "history": history,
                "impressions": impressions,
            })
    return behaviors


class MINDTrainDataset(Dataset):
    """Training dataset: each sample = (history, 1 positive + K negatives)."""

    def __init__(self, news, behaviors, his_size=50, npratio=4, max_title_len=30):
        self.news = news
        self.his_size = his_size
        self.npratio = npratio
        self.max_title_len = max_title_len

        self.samples = []
        for beh in behaviors:
            history = beh["history"]
            pos = []
            neg = []
            for imp in beh["impressions"]:
                nid, label = imp.rsplit("-", 1)
                if label == "1":
                    pos.append(nid)
                else:
                    neg.append(nid)
            for p in pos:
                self.samples.append((history, p, neg))

    def __len__(self):
        return len(self.samples)

    def _get_news_feature(self, nid):
        entry = self.news.get(nid, self.news["PAD"])
        return entry["input_ids"], entry["attention_mask"]

    def __getitem__(self, idx):
        history, pos_nid, neg_list = self.samples[idx]

        # Sample negatives
        if len(neg_list) >= self.npratio:
            neg_sampled = random.sample(neg_list, self.npratio)
        else:
            neg_sampled = neg_list + random.choices(neg_list, k=self.npratio - len(neg_list)) if neg_list else ["PAD"] * self.npratio

        # Candidate news: [positive, neg1, neg2, ...]
        candidates = [pos_nid] + neg_sampled
        cand_ids = np.stack([self._get_news_feature(n)[0] for n in candidates])
        cand_mask = np.stack([self._get_news_feature(n)[1] for n in candidates])

        # History: take last his_size clicks
        hist = history[-self.his_size:]
        clicked_mask = np.zeros(self.his_size, dtype=np.float32)
        hist_ids_list = []
        hist_mask_list = []
        for i in range(self.his_size):
            if i < len(hist):
                ids, mask = self._get_news_feature(hist[i])
                hist_ids_list.append(ids)
                hist_mask_list.append(mask)
                clicked_mask[i] = 1.0
            else:
                hist_ids_list.append(self.news["PAD"]["input_ids"])
                hist_mask_list.append(self.news["PAD"]["attention_mask"])

        hist_ids = np.stack(hist_ids_list)
        hist_mask = np.stack(hist_mask_list)

        # Label: first candidate is positive
        label = 0

        return {
            "candidate_input_ids": cand_ids.astype(np.int64),
            "candidate_attention_mask": cand_mask.astype(np.int64),
            "clicked_input_ids": hist_ids.astype(np.int64),
            "clicked_attention_mask": hist_mask.astype(np.int64),
            "clicked_mask": clicked_mask,
            "label": label,
        }


class MINDEvalDataset(Dataset):
    """Evaluation dataset: one sample per impression."""

    def __init__(self, news, behaviors, his_size=50, max_title_len=30):
        self.news = news
        self.his_size = his_size
        self.max_title_len = max_title_len
        self.behaviors = behaviors

        self.samples = []
        for beh in behaviors:
            history = beh["history"]
            impr_news = []
            labels = []
            for imp in beh["impressions"]:
                nid, label = imp.rsplit("-", 1)
                impr_news.append(nid)
                labels.append(int(label))
            if impr_news:
                self.samples.append((history, impr_news, labels, beh["impr_id"]))

    def __len__(self):
        return len(self.samples)

    def _get_news_feature(self, nid):
        entry = self.news.get(nid, self.news["PAD"])
        return entry["input_ids"], entry["attention_mask"]

    def __getitem__(self, idx):
        history, impr_news, labels, impr_id = self.samples[idx]

        # History
        hist = history[-self.his_size:]
        clicked_mask = np.zeros(self.his_size, dtype=np.float32)
        hist_ids_list = []
        hist_mask_list = []
        for i in range(self.his_size):
            if i < len(hist):
                ids, mask = self._get_news_feature(hist[i])
                hist_ids_list.append(ids)
                hist_mask_list.append(mask)
                clicked_mask[i] = 1.0
            else:
                hist_ids_list.append(self.news["PAD"]["input_ids"])
                hist_mask_list.append(self.news["PAD"]["attention_mask"])

        return {
            "clicked_input_ids": np.stack(hist_ids_list).astype(np.int64),
            "clicked_attention_mask": np.stack(hist_mask_list).astype(np.int64),
            "clicked_mask": clicked_mask,
            "impr_news": impr_news,
            "labels": labels,
            "impr_id": impr_id,
        }
