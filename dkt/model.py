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

        #0531 feature_embed : feature별 embedding 함수 dictionary
        #                     각 categorical feature의 가짓수를 불러와 각각의 embedding 함수를 선언합니다.
        #     cont_proj     : continuous feature의 linear 함수.
        #     comb_proj     : columns 수 + 1(interaction) 만큼의 hidden_dim -> hidden_dim
        self.feature_embed = {
            'interaction': nn.Embedding(3, self.hidden_dim // 3).to(self.args.device),
        }
        for feature in self.args.cate_cols:
            self.feature_embed[feature] = nn.Embedding(self.args.n_cate_size[feature]+1,
                                                       self.hidden_dim // 3).to(self.args.device)

        self.cont_proj = nn.Linear(1, self.hidden_dim//3)
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * (len(self.args.cate_cols) + len(self.args.cont_cols) + 1), self.hidden_dim)

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
        _, features, mask, interaction, _ = input

        batch_size = interaction.size(0)

        #0531 categorical : feature_embed를 순회하며 embedding을 호출하여 적용시킵니다.
        #     continuous  : cont_proj 통과시킵니다.
        #     각 output을 concat하여 comb_proj 통과시킵니다.
        embed_interaction = self.feature_embed['interaction'](interaction)
        embed = []
        for i, feature in enumerate(features):
            if i in self.args.cate_idx:
                embed.append(self.feature_embed[self.args.cate_idx[i]](feature))
            else:
                feature = feature.view(feature.size(0), -1, feature.size(-1))
                embed.append(self.cont_proj(feature.view(feature.size(0), self.args.max_seq_len, -1)))
        embed = torch.cat([embed_interaction] + embed, 2)
        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTM_merge(nn.Module):
    def __init__(self, args):
        super(LSTM_merge, self).__init__()
        self.args = args
        self.device = args.device

        #0531 total_cate_size : categorical feature의 모든 가짓수
        #     cate_cols       : categorical feature의 column명
        #     cont_cols       : continuous feature의 column명
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.total_cate_size = self.args.total_cate_size
        self.cate_cols = self.args.cate_cols
        self.cont_cols = self.args.cont_cols

        #0531 cate_emb  : categorical feature의 embedding 함수
        #     cate_proj : categorical embedding의 linear 함수. continuous feature가 없다면 hidden_dim*2, 그렇지않다면 hidden_dim
        #     cont_bn   : continuous feature의 batchnorm
        #     cont_emb  : continuous feature의 linear 함수. continuous feature의 수를 input. categorical feature가 없다면
        #                 hidden_dim*2, 그렇지않다면 hidden_dim
        #     comb_proj : hidden_dim*2 -> hidden_dim
        if len(self.cate_cols) != 0:
            self.cate_emb = nn.Embedding(self.total_cate_size, self.hidden_dim, padding_idx=0).to(self.args.device)
            if len(self.cont_cols) == 0:
                self.cate_proj = nn.Sequential(
                    nn.Linear(self.hidden_dim * len(self.cate_cols), self.hidden_dim*2),
                    nn.LayerNorm(self.hidden_dim*2)
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
                    nn.Linear(len(self.cont_cols), self.hidden_dim*2),
                    nn.LayerNorm(self.hidden_dim)
                )
            else:
                self.cont_emb = nn.Sequential(
                    nn.Linear(len(self.cont_cols), self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim)
                )

        self.comb_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
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
        #_, cate_x, cont_x, mask, interaction, _ = input
        cate_x, cont_x, mask, target = input

        cate_x = cate_x.to(self.args.device)
        cont_x = cont_x.to(self.args.device)
        mask = mask.to(self.args.device)

        batch_size = cate_x.size(0)
        seq_len = cate_x.size(1)

        #0531 categorical : cate_emb -> cate_proj
        #     continuous  : cont_bn -> cont_emb
        #     embedding별로 각각, 혹은 둘을 concat하여 seq_emb에 저장하여 comb_proj를 통과시킵니다.
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

        #mask, _ = mask.view(batch_size, seq_len, -1).max(2)


        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(seq_emb, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTMATTN(nn.Module):

    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding
        # 0526 maroo
        self.cate_embed = {
            'interaction': nn.Embedding(3, self.hidden_dim // 3).to(self.args.device)
        }
        for feature in self.args.n_features.keys():
            self.cate_embed[feature[2:]] = nn.Embedding(self.args.n_features[feature] + 1, self.hidden_dim // 3).to(
                self.args.device)
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * (len(self.args.n_features) + 1), self.hidden_dim)
        ###

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

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

        embed_interaction = self.cate_embed['interaction'](interaction)
        embed = torch.cat([embed_interaction] + [self.cate_embed[self.args.features_idx[i]](feature) for i, feature in
                                                 enumerate(features)], 2)
        ###

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        out = out.to('cuda')
        extended_attention_mask = extended_attention_mask.to('cuda')
        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        #0526 maroo
        self.cate_embed = {
            'interaction': nn.Embedding(3, self.hidden_dim // 3).to(self.args.device)
        }
        for feature in self.args.n_features.keys():
            self.cate_embed[feature[2:]] = nn.Embedding(self.args.n_features[feature]+1, self.hidden_dim // 3).to(self.args.device)
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * (len(self.args.n_features)+1), self.hidden_dim)
        ###


        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        #0526 maroo
        _, features, mask, interaction, _ = input

        batch_size = interaction.size(0)

        embed_interaction = self.cate_embed['interaction'](interaction)
        embed = torch.cat([embed_interaction]+[self.cate_embed[self.args.features_idx[i]](feature) for i, feature in enumerate(features)], 2)
        ###

        X = self.comb_proj(embed)

        # Bert
        X = X.to('cuda')
        mask = mask.to('cuda')
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds