import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from geoopt.manifolds.stereographic import math as math1
from transformers import AutoModel

class Scorer( nn.Module):
    def __init__(self, bert_model_path, vocab_size ,embed_dim = 768 ):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(bert_model_path)
        self.bert_model.resize_token_embeddings( vocab_size )
        self.ln_score = nn.Linear( embed_dim, 1 )

    def forward(self, inputs):
        ## inputs is 3-dimensional batch_size x 3 x seq_len 
        ## pair_masks shape:  batch_size x passage_pair
        ## input_ids, token_type_ids , attention_mask = inputs[:,0,:].contiguous(), inputs[:,1,:].contiguous(), inputs[:,2,:].contiguous()

        net = self.bert_model(**inputs)[0]
        ## CLS token's embedding
        net = net[:, 0, :].contiguous()
        score = F.sigmoid(self.ln_score(F.relu(net))).squeeze(1)
        return score


class Scorer_PER_v1(nn.Module):
    def __init__(self, bert_model_path, vocab_size, embed_dim=768):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(bert_model_path)
        self.bert_model.resize_token_embeddings(vocab_size)
        self.ln_score = nn.Linear(embed_dim + 1, 1)
        self.ln_cat = nn.Linear(embed_dim, 512)

    def forward(self, inputs):
        ## inputs is 3-dimensional batch_size x 3 x seq_len 
        ## pair_masks shape:  batch_size x passage_pair
        ## input_ids, token_type_ids , attention_mask = inputs[:,0,:].contiguous(), inputs[:,1,:].contiguous(), inputs[:,2,:].contiguous()

        net = self.bert_model(**inputs['param'])[0]

        citing_category_embed = inputs['category_batch_query'].to(torch.float32)
        citing_category_embed = self.ln_cat(citing_category_embed)
        candidate_category_embed = inputs['category_batch_candidate'].to(torch.float32)
        candidate_category_embed = self.ln_cat(candidate_category_embed)
        d = math1.dist(citing_category_embed, candidate_category_embed, k=torch.tensor(1.))

        ## CLS token's embedding
        net = net[:, 0, :].contiguous()
        final_emb = torch.cat((net, d.unsqueeze(1)), dim=1)
        score = F.sigmoid(self.ln_score(F.relu(final_emb))).squeeze(1)
        return score


class Scorer_PER_v2(nn.Module):
    def __init__(self, bert_model_path, vocab_size, embed_dim=768):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(bert_model_path)
        self.bert_model.resize_token_embeddings(vocab_size)
        self.ln_score = nn.Linear(embed_dim, 1)
        self.ln_cat = nn.Linear(embed_dim, 512)
        self.ln_out = nn.Linear(2, 1)

    def forward(self, inputs):
        ## inputs is 3-dimensional batch_size x 3 x seq_len 
        ## pair_masks shape:  batch_size x passage_pair
        ## input_ids, token_type_ids , attention_mask = inputs[:,0,:].contiguous(), inputs[:,1,:].contiguous(), inputs[:,2,:].contiguous()

        net = self.bert_model(**inputs['param'])[0]

        citing_category_embed = inputs['category_batch_query'].to(torch.float32)
        # print(citing_category_embed)
        # print(type(citing_category_embed))
        # print('Citing Category Embed Shape:', citing_category_embed.shape)
        citing_category_embed = self.ln_cat(citing_category_embed)
        candidate_category_embed = inputs['category_batch_candidate'].to(torch.float32)
        candidate_category_embed = self.ln_cat(candidate_category_embed)
        d = math1.dist(citing_category_embed, candidate_category_embed, k=torch.tensor(1.))

        ## CLS token's embedding
        net = net[:, 0, :].contiguous()
        similarity = F.sigmoid(self.ln_score(F.relu(net))).squeeze(1)

        similarity_column = similarity.unsqueeze(1)
        d_column = d.unsqueeze(1)
        concatenated_tensor = torch.cat((similarity_column, d_column), dim=1)
        score = F.sigmoid(self.ln_out(concatenated_tensor)).squeeze(1)
        return score
