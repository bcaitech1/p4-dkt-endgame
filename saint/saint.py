import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SAINTModel(nn.Module):

    def __init__(self, ninp:int=200, nhead:int=2, nhid:int=64, nlayers:int=2, dropout:float=0.3, device=device):
        super(TransformerModel, self).__init__()
        self.src_mask = None
        encoder_layers = TransformerEncoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout, activation='relu')
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=nlayers)
        self.interaction_embeddings = nn.Embedding(num_embeddings=3, embedding_dim=ninp)
        self.assessment_embeddings = nn.Embedding(num_embeddings=9455, embedding_dim=ninp)
        self.test_embeddings = nn.Embedding(num_embeddings=1538, embedding_dim=ninp)
        self.knowledge_embeddings = nn.Embedding(num_embeddings=913, embedding_dim=ninp)
        self.pos_embedding = nn.Embedding(200, ninp) # positional embeddings # 200 = max_len
        self.device = device
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, interaction, assessmentItemID, testId, KnowledgeTag, mask):
        embedded_src = self.interaction_embeddings(interaction) + \
                       self.assessment_embeddings(assessmentItemID) + \
                       self.test_embeddings(testId) + \
                       self.knowledge_embeddings(KnowledgeTag) + \
                       self.pos_embedding(torch.arange(0,assessmentItemID.shape[1]).to(self.device)).unsqueeze(0)
        embedded_src = embedded_src.transpose(0, 1)
        _src = embedded_src * np.sqrt(self.ninp)
        
        output = self.transformer_encoder(src=_src, src_key_padding_mask=mask)
        output = self.sigmoid(self.decoder(output))
        output = output.transpose(1, 0)
        return output