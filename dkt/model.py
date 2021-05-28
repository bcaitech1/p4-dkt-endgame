import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import copy
import math

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel    
except:
    from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel    



class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding 
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        # 0526 maroo
        self.feature_embed = {
            'interaction': nn.Embedding(3, self.hidden_dim // 3).to(self.args.device),
        }
        for feature in self.args.n_features.keys():
            if feature[-4:] == '_cat':
                self.feature_embed[feature[2:]] = nn.Embedding(self.args.n_features[feature] + 1,
                                                               self.hidden_dim // 3).to(self.args.device)
            elif feature[-4:] == '_con':
                self.feature_embed[feature[2:]] = nn.Embedding(self.args.n_features[feature] + 1,
                                                               self.hidden_dim // 3).to(self.args.device)
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * (len(self.args.n_features) + 1), self.hidden_dim)
        ###
        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):
        # 0526 maroo
        _, features, mask, interaction, _ = input

        batch_size = interaction.size(0)

        embed_interaction = self.feature_embed['interaction'](interaction)
        embed = torch.cat([embed_interaction] +
                          [self.feature_embed[self.args.features_idx[i]](feature) for i, feature in enumerate(features)], 2)
        ###
        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTM_0528(nn.Module):
    def __init__(self, args):
        super(LSTM_0528, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.total_cate_size = self.args.total_cate_size
        self.cate_cols = self.args.cate_cols
        self.cont_cols = self.args.cont_cols

        if len(self.cate_cols) != 0:
            self.cate_emb = nn.Embedding(self.total_cate_size, self.hidden_dim, padding_idx=0).to(self.args.device)
            if len(self.cont_cols) == 0:
                self.cate_proj = nn.Sequential(
                    nn.Linear(self.hidden_dim * len(self.cate_cols), self.hidden_dim * 2),
                    nn.LayerNorm(self.hidden_dim * 2)
                )
            else:
                self.cate_proj = nn.Sequential(
                    nn.Linear(self.hidden_dim * len(self.cate_cols), self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim)
                )

        if len(self.cont_cols) != 0:
            self.cont_bn = nn.BatchNorm1d(len(self.cont_cols))
            if len(self.cate_cols) == 0:
                self.cont_emb = nn.Sequential(
                    nn.Linear(len(self.cont_cols), self.hidden_dim * 2),
                    nn.LayerNorm(self.hidden_dim)
                )
            else:
                self.cont_emb = nn.Sequential(
                    nn.Linear(len(self.cont_cols), self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim)
                )

        self.comb_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        cate_x, cont_x, mask, _ = input

        cate_x = cate_x.to(self.args.device)
        cont_x = cont_x.to(self.args.device)
        mask = cont_x.to(self.args.device)

        batch_size = cate_x.size(0)
        seq_len = cate_x.size(1)

        if len(self.cate_cols) == 0:
            cont_x = self.cont_bn(cont_x.view(-1, cont_x.size(-1))).view(batch_size, -1, cont_x.size(-1))
            seq_emb = self.cont_emb(cont_x.view(batch_size, seq_len, -1))
        elif len(self.cont_cols) == 0:
            cate_emb = self.cate_emb(cate_x).view(batch_size, seq_len, -1)
            seq_emb = self.cate_proj(cate_emb)
        else:
            cate_emb = self.cate_emb(cate_x).view(batch_size, seq_len, -1)
            cate_emb = self.cate_proj(cate_emb)
            cont_x = self.cont_bn(cont_x.view(-1, cont_x.size(-1))).view(batch_size, -1, cont_x.size(-1))
            cont_emb = self.cont_emb(cont_x.view(batch_size, seq_len, -1))
            seq_emb = torch.cat([cate_emb, cont_emb], 2)

        seq_emb = self.comb_proj(seq_emb)

        # mask, _ = mask.view(batch_size, seq_len, -1).max(2)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(seq_emb, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds