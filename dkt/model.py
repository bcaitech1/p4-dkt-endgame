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