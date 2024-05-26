import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from src.prefetch.utils import ensure_dir_exists
from src.prefetch.rankers import PrefetchEncoder

# List of available datasets
datasets = ["arxiv", "acl", "refseer", "peerread"]

# Select a dataset and set enrichment flag
dataset_name = "refseer"
enrich = True

# Paths to the prefetch and reranker models for each dataset
prefetch_path = {
    "arxiv": "data_used_in_paper/model/prefetch/arxiv/model_batch_1645000.pt",
    "acl": "data_used_in_paper/model/prefetch/acl/model_batch_35000.pt",
    "peerread": "data_used_in_paper/model/prefetch/peerread/model_batch_10000.pt",
    "refseer": "data_used_in_paper/model/prefetch/refseer/model_batch_1195000.pt",
}

reranker_path = {
    "arxiv": "data_used_in_paper/model/rerank/arxiv/scibert/NN_prefetch/model_batch_190000.pt",
    "acl": "data_used_in_paper/model/rerank/acl/scibert/NN_prefetch/model_batch_91170.pt",
    "peerread": "data_used_in_paper/model/rerank/peerread/scibert/NN_prefetch/model_batch_28089.pt",
    "refseer": "data_used_in_paper/model/rerank/refseer/scibert/NN_prefetch/model_batch_60000.pt",
}

# Load dataset files
contexts = json.load(open(f"data_used_in_paper/data/{dataset_name}/contexts.json"))
papers = json.load(open(f"data_used_in_paper/data/{dataset_name}/papers.json"))
train_set = json.load(open(f"data_used_in_paper/data/{dataset_name}/train.json"))
val_set = json.load(open(f"data_used_in_paper/data/{dataset_name}/val.json"))
test_set = json.load(open(f"data_used_in_paper/data/{dataset_name}/test.json"))

# Get the model paths
prefetch_model_path = prefetch_path[dataset_name]
reranker_model_path = reranker_path[dataset_name]
paper_database_path = f"data_used_in_paper/data/{dataset_name}/papers.json"
unigram_words_path = "data_used_in_paper/model/glove/vocabulary_200dim.pkl"
prefetch_embedding_path = "data_used_in_paper/embedding/{dataset_name}/paper_embedding.pkl"


def compute_embeddings(paper_database_path, prefetch_model_path, unigram_words_path, prefetch_embedding_path,
                       embed_dim=200, encoder_gpu_list=[0], num_heads=8, hidden_dim=1024, max_seq_len=512,
                       max_doc_len=3, n_para_types=100, num_enc_layers=1, start=0, size=0, paragraphs_cache_size=16):
    import json
    import numpy as np
    import pickle
    from tqdm import tqdm

    # Load paper database
    paper_database = json.load(open(paper_database_path))

    # Get list of paper IDs and shuffle
    available_paper_ids = list(paper_database.keys())
    available_paper_ids.sort()
    np.random.shuffle(available_paper_ids)

    # Load the prefetch model
    ckpt_name = prefetch_model_path
    encoder = PrefetchEncoder(ckpt_name, unigram_words_path, embed_dim, encoder_gpu_list, num_heads, hidden_dim,
                              max_seq_len, max_doc_len, n_para_types, num_enc_layers)

    # Initialize variables for embeddings and mappers
    embeddings = []
    index_to_id_mapper = {}
    id_to_index_mapper = {}

    paragraphs_cache = []

    # Loop through papers and compute embeddings
    for index, paperid in enumerate(tqdm(available_paper_ids)):
        if index < start:
            continue
        if size > 0 and index >= start + size:
            break

        paper = paper_database[paperid]
        paragraphs = [[paper["title"], 0], [paper["abstract"], 1]]
        paragraphs_cache.append(paragraphs)
        if len(paragraphs_cache) >= paragraphs_cache_size:
            embeddings.append(encoder.encode(paragraphs_cache))
            paragraphs_cache = []

        index_to_id_mapper[index] = paperid
        id_to_index_mapper[paperid] = index

    # Encode remaining paragraphs in cache
    if len(paragraphs_cache) > 0:
        embeddings.append(encoder.encode(paragraphs_cache))
        paragraphs_cache = []

    if len(embeddings) > 0:
        embeddings = np.concatenate(embeddings, axis=0)
    else:
        embeddings = np.zeros((0, embed_dim)).astype(np.float32)

    # Save embeddings to file
    with open(ensure_dir_exists(prefetch_embedding_path), "wb") as f:
        pickle.dump({
            "index_to_id_mapper": index_to_id_mapper,
            "id_to_index_mapper": id_to_index_mapper,
            "embedding": embeddings
        }, f, -1)


# Compute embeddings if not already present
if not os.path.exists(prefetch_embedding_path):
    compute_embeddings(paper_database_path, prefetch_model_path, unigram_words_path, prefetch_embedding_path)

# Create a directed graph from the contexts
G = nx.DiGraph()

for context in contexts:
    # Extract citing and cited paper IDs
    citing, cited = context.split("_")[:2]
    # Add nodes if they don't exist
    if citing not in G.nodes:
        G.add_node(context)
    if cited not in G.nodes:
        G.add_node(context)
    G.add_edge(citing, cited)

# Calculate and print average degree
avg_degree = sum(dict(G.degree()).values()) / len(G)
print("Average Degree:", avg_degree)

# Calculate and print local clustering coefficients
local_clustering_coefficients = nx.average_clustering(G)
print("Local Clustering Coefficients:", local_clustering_coefficients)


def remove_duplicates(candidate_list):
    """Removes duplicates from a list without changing the order."""
    seen = set()
    new_list = []
    for item in candidate_list:
        if item not in seen:
            seen.add(item)
            new_list.append(item)
    return new_list


def enrich(candidate_list):
    """Enriches the candidate list by adding references of the papers in the list."""
    original_list = candidate_list.copy()
    for paperid in original_list:
        for _, ref_paper in G.out_edges(paperid):
            candidate_list.append(ref_paper)
    return remove_duplicates(candidate_list)


from citation_recommender import *

prefetcher = Prefetcher(
    model_path=prefetch_model_path,
    embedding_path=prefetch_embedding_path,
    gpu_list=[0]
)
reranker = Reranker(model_path=reranker_model_path, gpu_list=[0])

# Initialize lists for evaluation metrics
hit_list = []
rank_list = []
reranked_candidates = []
top_K = 51

# Loop through test set and perform prefetching and reranking
for idx in tqdm(range(len(test_set))):
    context_info = contexts[test_set[idx]["context_id"]]
    citing_id = context_info["citing_id"]
    refid = context_info["refid"]  # The ground-truth cited paper

    local_context = context_info["masked_text"]
    citing_paper = papers[citing_id]
    citing_title = citing_paper["title"]
    citing_abstract = citing_paper["abstract"]

    # Get top 100 candidates from prefetcher
    candi_list = prefetcher.get_top_n(
        {
            "citing_title": citing_title,
            "citing_abstract": citing_abstract,
            "local_context": local_context
        }, 100  # 100 candidates
    )

    # Enrich the candidate list if needed
    if enrich:
        candi_list = enrich(candi_list)

    candidate_list = [{
        "paper_id": pid,
        "title": papers[pid].get("title", ""),
        "abstract": papers[pid].get("abstract", "")
    } for pid in candi_list]

    # Rerank the candidate list
    reranked_candidate_list = reranker.rerank(citing_title, citing_abstract, local_context, candidate_list)
    reranked_candidate_ids = [item["paper_id"] for item in reranked_candidate_list]
    reranked_candidates.append(reranked_candidate_ids)

    # Calculate rank and hits
    if refid not in reranked_candidate_ids:
        rank_list.append(100)
    else:
        rank_list.append(reranked_candidate_ids.index(refid) + 1)

from math import log2


def mrr(ranks, k):
    """Calculate Mean Reciprocal Rank (MRR) at rank k."""
    rec_ranks = [1. / r if r <= k else 0. for r in ranks]
    return sum(rec_ranks) / len(ranks)


def recall(ranks, k):
    """Calculate Recall at rank k."""
    return sum(r <= k for r in ranks) / len(ranks)


def ndcg(ranks, k):
    """Calculate Normalized Discounted Cumulative Gain (NDCG) at rank k."""
    ndcg_per_query = sum(1 / log2(r + 1) for r in ranks if r <= k)
    return ndcg_per_query / len(ranks)


# Print evaluation metrics
print("Recall@5: ", recall(rank_list, 5))
print("Recall@10: ", recall(rank_list, 10))
print("Recall@20: ", recall(rank_list, 20))
print("Recall@50: ", recall(rank_list, 50))
print("NDCG@10: ", ndcg(rank_list, 10))
print("MRR@10: ", mrr(rank_list, 10))
