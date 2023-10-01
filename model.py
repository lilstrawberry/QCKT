import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import glo

def attention_score(query, key, value, mask, gamma):
    # batch head seq seq
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])

    seq = scores.shape[-1]
    x1 = torch.arange(seq).float().unsqueeze(-1).to(query.device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask, -1e9)
        scores_ = torch.softmax(scores_, dim=-1)

        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        position_effect = torch.abs(x1 - x2)[None, None, :, :]  # 1 1 seq seq
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.0
        )
        dist_scores = dist_scores.sqrt().detach()

    gamma = -1.0 * gamma.abs().unsqueeze(0)  # 1 head 1 1
    total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)

    scores = scores * total_effect
    scores = torch.masked_fill(scores, mask, -1e9)
    scores = torch.softmax(scores, dim=-1)
    scores = torch.masked_fill(scores, mask, 0)

    output = torch.matmul(scores, value)

    return output, scores

class MultiHead_Forget_Attn(nn.Module):
    def __init__(self, d, p, head):
        super(MultiHead_Forget_Attn, self).__init__()

        self.q_linear = nn.Linear(d, d)
        self.k_linear = nn.Linear(d, d)
        self.v_linear = nn.Linear(d, d)
        self.linear_out = nn.Linear(d, d)
        self.head = head
        self.gammas = nn.Parameter(torch.zeros(head, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

    def forward(self, query, key, value, mask):
        # query: batch seq d
        batch = query.shape[0]
        origin_d = query.shape[-1]
        d_k = origin_d // self.head
        query = self.q_linear(query).view(batch, -1, self.head, d_k).transpose(1, 2)
        key = self.k_linear(key).view(batch, -1, self.head, d_k).transpose(1, 2)
        value = self.v_linear(value).view(batch, -1, self.head, d_k).transpose(1, 2)
        out, attn = attention_score(query, key, value, mask, self.gammas)
        # out, attn = getAttention(query, key, value, mask)
        # batch head seq d_k
        out = out.transpose(1, 2).contiguous().view(batch, -1, origin_d)
        out = self.linear_out(out)
        return out, attn

class TransformerLayer(nn.Module):
    def __init__(self, d, p, head):
        super(TransformerLayer, self).__init__()

        self.dropout = nn.Dropout(p)

        self.linear1 = nn.Linear(d, d)
        self.linear2 = nn.Linear(d, d)
        self.layer_norm1 = nn.LayerNorm(d)
        self.layer_norm2 = nn.LayerNorm(d)
        self.activation = nn.ReLU()
        self.attn = MultiHead_Forget_Attn(d, p, head)

    def forward(self, q, k, v, mask):
        out, _ = self.attn(q, k, v, mask)
        q = q + self.dropout(out)
        q = self.layer_norm1(q)
        query2 = self.linear2(self.dropout(self.activation(self.linear1(q))))
        q = q + self.dropout((query2))
        q = self.layer_norm2(q)
        return q

class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim, p):
        super(GCNConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w = nn.Parameter(torch.rand((in_dim, out_dim)))
        nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(torch.rand((out_dim)))
        nn.init.zeros_(self.b)

        self.dropout = nn.Dropout(p=p)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = torch.matmul(x, self.w)
        x = torch.sparse.mm(adj.float(), x)
        x = x + self.b
        return x

def bt_loss(h1, h2, batch_norm=True, eps=1e-15):
    batch_size = h1.size(0)
    feature_dim = h1.size(1)

    lambda_ = 1. / feature_dim
    z1_norm = (h1 - h1.mean(dim=0)) / (h1.std(dim=0) + eps)
    z2_norm = (h2 - h2.mean(dim=0)) / (h2.std(dim=0) + eps)
    c = (z1_norm.T @ z2_norm) / batch_size
    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (1 - c.diagonal()).pow(2).sum()
    loss += lambda_ * c[off_diagonal_mask].pow(2).sum()
    return loss

def sim(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    z = torch.matmul(x, y.transpose(-1, -2))
    tau = 0.8
    return torch.exp(z / tau)

def batched_semi_loss(z1, z2, batch_size):
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    indices = torch.arange(0, num_nodes).to(device)
    losses = []

    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]
        refl_sim = sim(z1[mask], z1)  # [B, N]
        between_sim = sim(z1[mask], z2)  # [B, N]

        losses.append(-torch.log(
            between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
            / (refl_sim.sum(1) + between_sim.sum(1)
               - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

    return torch.cat(losses).mean()

def unself_loss(x, y):
    return batched_semi_loss(x, y, 10000)
    # + batched_semi_loss(y, x, 10000)

def self_loss(z1, z2, true_matrix):
    device = z1.device
    num_nodes = z1.size(0)

    now_use_batch = 10000

    num_batches = (num_nodes - 1) // now_use_batch + 1
    indices = torch.arange(0, num_nodes).to(device)
    losses = []
    now_matrix = true_matrix.to_dense()

    for i in range(num_batches):
        mask = indices[i * now_use_batch:(i + 1) * now_use_batch]
        refl_sim = sim(z1[mask], z1)  # [B, N]
        between_sim = sim(z1[mask], z2)  # [B, N]
        batch_true_matrix = now_matrix[mask]  # [B, N]

        fenzi = (refl_sim * batch_true_matrix).sum(dim=-1) + (between_sim * batch_true_matrix).sum(dim=-1)
        fenmu = refl_sim.sum(dim=-1) + between_sim.sum(dim=-1)

        losses.append(-torch.log((fenzi + 1e-8) / fenmu))

    return torch.cat(losses).mean()

def self_loss_1(x, y):
    return self_loss(x, y, glo.get_value('pos_matrix'))
    # + self_loss(y, x, pos_matrix)

def self_loss_2(x, y):
    return self_loss(x, y, glo.get_value('unique_pos_matrix'))
    # + self_loss(y, x, unique_pos_matrix)

class DKT_QCKT(nn.Module):
    def __init__(self, pro_max, skill_max, d, p, phi):
        super(DKT_QCKT, self).__init__()

        self.skill_max = skill_max
        self.phi = phi

        self.gcn = GCNConv(d, d, p)

        self.pro_embed = nn.Parameter(torch.rand(pro_max, d))

        self.skill_embed = nn.Parameter(torch.rand(skill_max, d))

        self.ans_embed = nn.Parameter(torch.rand(2, d))

        self.lstm = nn.LSTM(d, d, batch_first=True)

        self.out = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(d, 1)
        )
        self.akt_pro_diff = nn.Parameter(torch.rand(pro_max, 1))
        self.akt_pro_change = nn.Parameter(torch.rand(pro_max, d))

        self.multi_head = nn.MultiheadAttention(d, 8, p)
        self.dropout = nn.Dropout(p=p)

    def get_rasch_embed(self):

        skill_contain, _ = self.multi_head(self.pro_embed, self.skill_embed, self.skill_embed,
                                           attn_mask=(glo.get_value('pro2skill') == 0))

        return skill_contain + self.akt_pro_diff * self.akt_pro_change

    def get_gcn_repre(self):
        return self.pro_embed + self.gcn(self.pro_embed, glo.get_value('gcn_matrix'))

    def forward(self, last_problem, last_ans, next_problem):

        rasch_embed = self.get_rasch_embed()
        gcn_embed = self.get_gcn_repre()

        contrast_loss = bt_loss(rasch_embed, gcn_embed) * self.phi

        # contrast_loss = unself_loss(rasch_embed, gcn_embed) * 0.01
        # contrast_loss = self_loss_1(rasch_embed, gcn_embed) * 0.01
        # contrast_loss = self_loss_2(rasch_embed, gcn_embed) * 0.01
        # contrast_loss = 0

        pro_embed = rasch_embed
        next_pro_embed = F.embedding(next_problem, pro_embed)

        ls_X, _ = self.lstm(
            self.dropout(F.embedding(last_problem, pro_embed) + F.embedding(last_ans.long(), self.ans_embed)))

        P = torch.sigmoid(self.out(self.dropout(torch.cat([ls_X, next_pro_embed], dim=-1)))).squeeze(-1)

        return P, contrast_loss