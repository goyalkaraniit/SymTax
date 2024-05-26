from src.prefetch.rankers import PrefetchEncoder
from src.prefetch.rankers import Ranker as PrefetchRanker
import pickle
import json
import numpy as np
from tqdm import tqdm

from src.rerank.datautils import RerankDataset
from src.rerank.model import Scorer, Scorer_PER_v1, Scorer_PER_v2

from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

class Prefetcher:
    def __init__( self,  model_path, 
                   embedding_path,
                   gpu_list = [],
                   vocab_path = "model/glove/vocabulary_200dim.pkl",
                   embed_dim = 200,
                   num_heads = 8,
                   hidden_dim = 1024,
                   max_seq_len = 512,
                   max_doc_len = 3,
                   n_para_types = 100,
                   num_enc_layers = 1,
                   
                   citation_title_label = 0,
                   citation_abstract_label = 1,
                   citation_context_label = 3,
                ):
        
        with open( vocab_path,"rb") as f:
            words = pickle.load(f)
        
        encoder = PrefetchEncoder( model_path, vocab_path, 
                                       embed_dim, gpu_list[:1] ,
                                       num_heads, hidden_dim,  
                                       max_seq_len, max_doc_len, 
                                       n_para_types, num_enc_layers
                                     )
        ranker = PrefetchRanker( embedding_path, embed_dim , gpu_list =  gpu_list  )
        ranker.encoder = encoder
        
        self.ranker = ranker
        self.encoder = encoder
        
        self.citation_title_label = citation_title_label
        self.citation_abstract_label = citation_abstract_label
        self.citation_context_label = citation_context_label
    
    def get_top_n( self, query, n = 10 ):
        """
          query structure 
           {
              "citing_title": The title of the citing paper, default = "" 
              "citing_abstract": The abstract of the citing paper, default = ""
              "local_context": The local citation sentence as the local context
           }
        """ 
        query_text = [
                        [ 
                            [ query.get("citing_title",""), self.citation_title_label ],
                            [ query.get("citing_abstract",""), self.citation_abstract_label ],
                            [ query.get("local_context", "") , self.citation_context_label ]  
                        ]  
                    ]
        candidates = self.ranker.get_top_n( n, query_text )
        return candidates


class Reranker:
    def __init__(self, 
                 model_path,
                 gpu_list = [],
                 initial_model_path = "allenai/scibert_scivocab_uncased" ):
        self.tokenizer = AutoTokenizer.from_pretrained( initial_model_path )
        self.tokenizer.add_special_tokens( { 'additional_special_tokens': ['<cit>','<sep>','<eos>'] } )
        vocab_size = len( self.tokenizer )
        
        ckpt = torch.load( model_path,  map_location=torch.device('cpu') )
        self.scorer = Scorer( initial_model_path, vocab_size )
        self.scorer.load_state_dict( ckpt["scorer"] )
        
        self.device = torch.device( "cuda:%d"%(gpu_list[0]) if torch.cuda.is_available() and len(gpu_list) > 0 else "cpu"  )
        self.scorer = self.scorer.to(self.device)

        if self.device.type == "cuda" and len( gpu_list ) > 1:
            self.scorer = DataParallel(self.scorer, gpu_list)
            
        self.sep_token = "<sep>"
    
    def rerank(self, citing_title = "", citing_abstract="", local_context="", original_candidate_list=[ {} ], max_input_length = 512, reranking_batch_size = 50 ):
        candidate_list = original_candidate_list.copy()
        if len(candidate_list) == 0:
            return []

        global_context = citing_title + " " + citing_abstract

        #  add section in query_text
        query_text = " ".join(global_context.split()[:int(max_input_length * 0.35)]) + self.sep_token + local_context
        score_list = []
        for pos in range(0, len(candidate_list), reranking_batch_size):
            candidate_batch = candidate_list[pos: pos + reranking_batch_size]
            query_text_batch = [query_text for _ in range(len(candidate_batch))]
            candidate_text_batch = [item.get("title", "") + " " + item.get("abstract", "") for item in candidate_batch]

            encoded_seqs = self.tokenizer(query_text_batch, candidate_text_batch, max_length=max_input_length,
                                          padding="max_length", truncation=True)
            for key in encoded_seqs:
                encoded_seqs[key] = torch.from_numpy(np.asarray(encoded_seqs[key])).to(self.device)

            with torch.no_grad():
                score_list.append(self.scorer({
                    "input_ids": encoded_seqs["input_ids"],
                    "token_type_ids": encoded_seqs["token_type_ids"],
                    "attention_mask": encoded_seqs["attention_mask"]
                }).detach())
        score_list = torch.cat(score_list, dim=0).view(-1).cpu().numpy().tolist()

        candidate_list, _ = list(zip(*sorted(zip(candidate_list, score_list), key=lambda x: -x[1])))

        return candidate_list


class Reranker_PER_v1:
    def __init__(self,
                 model_path,
                 gpu_list=[],
                 initial_model_path="allenai/scibert_scivocab_uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(initial_model_path)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<cit>', '<sep>', '<eos>']})
        vocab_size = len(self.tokenizer)

        ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        self.scorer = Scorer_PER_v1(initial_model_path, vocab_size)
        self.scorer.load_state_dict(ckpt["scorer"])

        self.device = torch.device(
            "cuda:%d" % (gpu_list[0]) if torch.cuda.is_available() and len(gpu_list) > 0 else "cpu")
        self.scorer = self.scorer.to(self.device)

        if self.device.type == "cuda" and len(gpu_list) > 1:
            self.scorer = DataParallel(self.scorer, gpu_list)

        self.sep_token = "<sep>"

    def rerank(self, citing_title="", citing_abstract="", local_context="", citing_category="",
               original_candidate_list=[{}], cat_map={}, max_input_length=512, reranking_batch_size=50, heading=""):
        candidate_list = original_candidate_list.copy()
        if len(candidate_list) == 0:
            return []

        global_context = citing_title + " " + citing_abstract

        #  add section in query_text
        query_text = " ".join(global_context.split()[:int(max_input_length * 0.35)]) + self.sep_token + local_context
        score_list = []
        for pos in range(0, len(candidate_list), reranking_batch_size):
            candidate_batch = candidate_list[pos: pos + reranking_batch_size]
            query_text_batch = [query_text for _ in range(len(candidate_batch))]
            candidate_text_batch = [item.get("title", "") + " " + item.get("abstract", "") for item in candidate_batch]

            encoded_seqs = self.tokenizer(query_text_batch, candidate_text_batch, max_length=max_input_length,
                                          padding="max_length", truncation=True)
            for key in encoded_seqs:
                encoded_seqs[key] = torch.from_numpy(np.asarray(encoded_seqs[key])).to(self.device)

            citing_category_batch = [cat_map[citing_category][0] for _ in range(len(candidate_batch))]
            citing_category_batch = torch.from_numpy(np.array(citing_category_batch)).to(self.device)
            candidate_category_batch = [cat_map[item['categories'][0]][0] for item in candidate_batch]
            candidate_category_batch = torch.from_numpy(np.array(candidate_category_batch)).to(self.device)

            with torch.no_grad():
                param = {
                    "input_ids": encoded_seqs["input_ids"],
                    "token_type_ids": encoded_seqs["token_type_ids"],
                    "attention_mask": encoded_seqs["attention_mask"]
                }
                score_list.append(self.scorer({'param': param, 'category_batch_query': citing_category_batch,
                                               'category_batch_candidate': candidate_category_batch}).detach())
        score_list = torch.cat(score_list, dim=0).view(-1).cpu().numpy().tolist()

        candidate_list, _ = list(zip(*sorted(zip(candidate_list, score_list), key=lambda x: -x[1])))

        return candidate_list


class Reranker_PER_v2:
    def __init__(self,
                 model_path,
                 gpu_list=[],
                 initial_model_path="allenai/scibert_scivocab_uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(initial_model_path)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<cit>', '<sep>', '<eos>']})
        vocab_size = len(self.tokenizer)

        ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        self.scorer = Scorer_PER_v2(initial_model_path, vocab_size)
        self.scorer.load_state_dict(ckpt["scorer"])

        self.device = torch.device(
            "cuda:%d" % (gpu_list[0]) if torch.cuda.is_available() and len(gpu_list) > 0 else "cpu")
        self.scorer = self.scorer.to(self.device)

        if self.device.type == "cuda" and len(gpu_list) > 1:
            self.scorer = DataParallel(self.scorer, gpu_list)

        self.sep_token = "<sep>"

    def rerank(self, citing_title="", citing_abstract="", local_context="", citing_category="",
               original_candidate_list=[{}], cat_map={}, max_input_length=512, reranking_batch_size=50, heading=""):
        candidate_list = original_candidate_list.copy()
        if len(candidate_list) == 0:
            return []

        global_context = citing_title + " " + citing_abstract

        #  add section in query_text
        query_text = " ".join(
            global_context.split()[:int(max_input_length * 0.35)]) + self.sep_token + local_context + heading
        score_list = []
        for pos in range(0, len(candidate_list), reranking_batch_size):
            candidate_batch = candidate_list[pos: pos + reranking_batch_size]
            query_text_batch = [query_text for _ in range(len(candidate_batch))]
            candidate_text_batch = [item.get("title", "") + " " + item.get("abstract", "") for item in candidate_batch]

            encoded_seqs = self.tokenizer(query_text_batch, candidate_text_batch, max_length=max_input_length,
                                          padding="max_length", truncation=True)
            for key in encoded_seqs:
                encoded_seqs[key] = torch.from_numpy(np.asarray(encoded_seqs[key])).to(self.device)

            citing_category_batch = [cat_map[citing_category][0] for _ in range(len(candidate_batch))]
            citing_category_batch = torch.from_numpy(np.array(citing_category_batch)).to(self.device)
            candidate_category_batch = [cat_map[item['categories'][0]][0] for item in candidate_batch]
            candidate_category_batch = torch.from_numpy(np.array(candidate_category_batch)).to(self.device)

            with torch.no_grad():
                param = {
                    "input_ids": encoded_seqs["input_ids"],
                    "token_type_ids": encoded_seqs["token_type_ids"],
                    "attention_mask": encoded_seqs["attention_mask"]
                }
                score_list.append(self.scorer({'param': param, 'category_batch_query': citing_category_batch,
                                               'category_batch_candidate': candidate_category_batch}).detach())
        score_list = torch.cat(score_list, dim=0).view(-1).cpu().numpy().tolist()

        candidate_list, _ = list(zip(*sorted(zip(candidate_list, score_list), key=lambda x: -x[1])))

        return candidate_list
