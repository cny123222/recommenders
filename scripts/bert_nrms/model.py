"""
BERT-NRMS: Neural News Recommendation with Multi-Head Self-Attention
using BERT as the news encoder instead of GloVe embeddings.

Architecture:
  News Encoder:  title tokens → BERT → last_hidden_state → Additive Attention → news_vec
  User Encoder:  [clicked news_vecs] → Multi-Head Self-Attention → Additive Attention → user_vec
  Score:         dot(user_vec, candidate_news_vec)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class AdditiveAttention(nn.Module):
    """Additive attention to aggregate a sequence into a single vector."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        attn = torch.tanh(self.proj(x))
        attn = self.query(attn).squeeze(-1)
        if mask is not None:
            bool_mask = mask.bool()
            all_masked = ~bool_mask.any(dim=-1, keepdim=True)
            safe_mask = bool_mask | all_masked
            attn = attn.masked_fill(~safe_mask, -1e4)
        attn = F.softmax(attn, dim=-1)
        return torch.bmm(attn.unsqueeze(1), x).squeeze(1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, mask=None):
        if mask is not None:
            bool_mask = mask.bool()
            all_masked = ~bool_mask.any(dim=-1, keepdim=True)
            safe_mask = bool_mask | all_masked
            kpm = ~safe_mask
        else:
            kpm = None
        out, _ = self.attn(x, x, x, key_padding_mask=kpm)
        return out


class NewsEncoder(nn.Module):
    """Encode a news title using BERT + additive attention."""

    def __init__(self, bert_model_name, news_dim, freeze_bert_layers=8):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size

        # Freeze early BERT layers for efficiency
        if freeze_bert_layers > 0:
            for name, param in self.bert.named_parameters():
                if "embeddings" in name:
                    param.requires_grad = False
                elif "encoder.layer." in name:
                    layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                    if layer_num < freeze_bert_layers:
                        param.requires_grad = False

        self.attention = AdditiveAttention(bert_hidden, news_dim)
        self.output_proj = nn.Linear(bert_hidden, news_dim) if bert_hidden != news_dim else nn.Identity()
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        # input_ids: (batch, seq_len)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.dropout(bert_out.last_hidden_state)  # (batch, seq_len, hidden)
        news_vec = self.attention(hidden, mask=attention_mask.bool())
        news_vec = self.output_proj(news_vec)
        return news_vec  # (batch, news_dim)


class UserEncoder(nn.Module):
    """Encode user from clicked news representations using self-attention."""

    def __init__(self, news_dim, num_heads, attention_hidden_dim):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(news_dim, num_heads)
        self.attention = AdditiveAttention(news_dim, attention_hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, clicked_news_vecs, clicked_mask=None):
        # clicked_news_vecs: (batch, his_size, news_dim)
        y = self.self_attn(clicked_news_vecs, mask=clicked_mask)
        y = self.dropout(y)
        user_vec = self.attention(y, mask=clicked_mask)
        return user_vec  # (batch, news_dim)


class BertNRMS(nn.Module):
    """BERT-NRMS model for news recommendation."""

    def __init__(self, bert_model_name="bert-base-uncased", news_dim=256,
                 num_heads=16, attention_hidden_dim=200, freeze_bert_layers=8):
        super().__init__()
        self.news_encoder = NewsEncoder(bert_model_name, news_dim, freeze_bert_layers)
        self.user_encoder = UserEncoder(news_dim, num_heads, attention_hidden_dim)
        self.news_dim = news_dim

    def encode_news(self, input_ids, attention_mask):
        return self.news_encoder(input_ids, attention_mask)

    def encode_user(self, clicked_news_vecs, clicked_mask=None):
        return self.user_encoder(clicked_news_vecs, clicked_mask)

    def forward(self, candidate_input_ids, candidate_attention_mask,
                clicked_input_ids, clicked_attention_mask, clicked_mask):
        """
        Args:
            candidate_input_ids: (batch, num_candidates, seq_len)
            candidate_attention_mask: (batch, num_candidates, seq_len)
            clicked_input_ids: (batch, his_size, seq_len)
            clicked_attention_mask: (batch, his_size, seq_len)
            clicked_mask: (batch, his_size) — which history slots are real vs padding
        Returns:
            logits: (batch, num_candidates)
        """
        batch_size = candidate_input_ids.size(0)
        num_candidates = candidate_input_ids.size(1)
        his_size = clicked_input_ids.size(1)
        seq_len = candidate_input_ids.size(2)

        # Encode all candidates
        cand_ids = candidate_input_ids.view(-1, seq_len)
        cand_mask = candidate_attention_mask.view(-1, seq_len)
        cand_vecs = self.encode_news(cand_ids, cand_mask)
        cand_vecs = cand_vecs.view(batch_size, num_candidates, -1)

        # Encode all clicked news
        click_ids = clicked_input_ids.view(-1, seq_len)
        click_mask = clicked_attention_mask.view(-1, seq_len)
        click_vecs = self.encode_news(click_ids, click_mask)
        click_vecs = click_vecs.view(batch_size, his_size, -1)

        # Encode user
        user_vecs = self.encode_user(click_vecs, clicked_mask)  # (batch, news_dim)

        # Score: dot product
        logits = torch.bmm(cand_vecs, user_vecs.unsqueeze(-1)).squeeze(-1)
        return logits  # (batch, num_candidates)
