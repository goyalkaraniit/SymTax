import json
import networkx as nx
from citation_recommender import *
import random
from tqdm import tqdm
from math import log2

# Configuration
dataset_name = "arxiv"
version = "v1"  # select from v1 and v2
llm = "scibert"  # select from {scibert, specter}
section = False  # select from {True, False}
suf = "_sec" if section else ""
enrich = True  # select from {}

# Paths
prefetch_model_path = f"model/prefetch/arxiv/model_batch_105000.pt"
paper_database_path = f"data_used_in_paper/data/{dataset_name}/papers.json"
unigram_words_path = "data_used_in_paper/model/glove/vocabulary_200dim.pkl"
prefetch_embedding_path = f"embedding/prefetch_arxiv/custom/paper_embedding.pkl"
reranker_model_path = f'model/rerank/arxiv/scibert/NN_prefetch{suf}/per_{version}/text/{llm}/model_batch_10000.pt'
GPU_NO = 0

# Load taxonomy mapping
with open('arysta/taxonomy_fused/Specter_data_text.json') as f:
    cat_map = json.load(f)

# Initialize prefetcher and reranker
prefetcher = Prefetcher(
    model_path=prefetch_model_path,
    embedding_path=prefetch_embedding_path,
    gpu_list=[GPU_NO]
)
specter_path = "allenai/scibert_scivocab_uncased"
reranker = Reranker_PER_v1(model_path=reranker_model_path, gpu_list=[GPU_NO])

# Load datasets
contexts = json.load(open(f"{dataset_name}/contexts_new_head.json"))
papers = json.load(open(f"{dataset_name}/papers_cat.json"))
train_set = json.load(open(f"{dataset_name}/train_data.json"))
val_set = json.load(open(f"{dataset_name}/val_data.json"))
test_set = json.load(open(f"{dataset_name}/test_data.json"))

# Build citation graph
G = nx.DiGraph()
for context in contexts:
    citing, cited = context.split("_")[:2]
    if citing not in G.nodes:
        G.add_node(context)
    if cited not in G.nodes:
        G.add_node(context)
    G.add_edge(citing, cited)


# Helper function to sort and enrich candidate list
def unique_sort_by_frequency_with_tiebreaker(input_list):
    frequency_dict = {}
    for i, item in enumerate(input_list):
        if item in frequency_dict:
            frequency_dict[item][0] += 1
        else:
            frequency_dict[item] = [1, i]  # [Frequency, Initial Index]
    sorted_unique_elements = sorted(frequency_dict.items(), key=lambda x: (-x[1][0], x[1][1]))
    return [item for item, freq_index in sorted_unique_elements]


def enrich_300(candidate_list):
    new_list = []
    for paperid in candidate_list:
        for _, ref_paper in G.out_edges(paperid):
            new_list.append(ref_paper)
    return candidate_list + unique_sort_by_frequency_with_tiebreaker(new_list)[:200]


# Evaluation metrics
def mrr(ranks, k):
    rec_ranks = [1. / r if r <= k else 0. for r in ranks]
    return sum(rec_ranks) / len(ranks)


def recall(ranks, k):
    return sum(r <= k for r in ranks) / len(ranks)


def ndcg(ranks, k):
    return sum(1 / log2(r + 1) for r in ranks if r <= k) / len(ranks)


# Main evaluation loop
hit_list = []
rank_list = []
reranked_candidates = []
top_K = 51

random.seed(12)
random_values = [random.randint(0, 100000) for _ in range(10000)]

for idx in tqdm(range(len(test_set))):
    context_info = contexts[test_set[idx]["context_id"]]
    citing_id = context_info["citing_id"]
    refid = context_info["ref_id"]  # Ground-truth cited paper

    local_context = context_info["masked_text"]
    citing_paper = papers[citing_id]
    citing_title = citing_paper["title"]
    citing_abstract = citing_paper["abstract"]
    citing_category = citing_paper["categories"][0]
    heading = context_info["heading"]

    # Prefetch top 100 candidates
    candi_list = prefetcher.get_top_n(
        {
            "citing_title": citing_title,
            "citing_abstract": citing_abstract,
            "local_context": local_context
        }, 100
    )

    # Enrich candidate list if required
    if enrich:
        candi_list = enrich_300(candi_list)

    candidate_list = [{
        "paper_id": pid,
        "title": papers[pid].get("title", ""),
        "categories": papers[pid].get("categories", ""),
        "abstract": papers[pid].get("abstract", "")
    } for pid in candi_list]

    # Rerank candidates
    reranked_candidate_list = reranker.rerank(
        citing_title, citing_abstract, local_context, citing_category,
        candidate_list, cat_map, heading=heading
    )
    reranked_candidate_ids = [item["paper_id"] for item in reranked_candidate_list]
    reranked_candidates.append(reranked_candidate_ids)

    # Update rank list
    if refid not in reranked_candidate_ids:
        rank_list.append(100)
    else:
        rank_list.append(reranked_candidate_ids.index(refid) + 1)

# Calculate and print metrics
print("Recall@5: ", recall(rank_list, 5))
print("Recall@10: ", recall(rank_list, 10))
print("Recall@20: ", recall(rank_list, 20))
print("Recall@50: ", recall(rank_list, 50))
print("NDCG@10: ", ndcg(rank_list, 10))
print("MRR@10: ", mrr(rank_list, 10))
