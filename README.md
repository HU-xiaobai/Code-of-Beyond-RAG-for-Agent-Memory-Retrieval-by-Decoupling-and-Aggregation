# xMemory

## Introduction

This repository accompanies our arxiv 2026.02 paper: 

**Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation**  
Zhanghao Hu, Qinglin Zhu, Hanqi Yan, Yulan He, Lin Gui  

📄 [Paper on Huggingface](https://huggingface.co/papers/2602.02007) | 📄 [arXiv Preprint](https://arxiv.org/abs/2602.02007) 🔥[Project Page](https://zhanghao-xmemory.github.io/Academic-project-page-template/)

> Agent memory systems often adopt the standard Retrieval-Augmented Generation (RAG) pipeline, yet its underlying assumptions differ in this setting. RAG targets large, heterogeneous corpora where retrieved passages are diverse, whereas agent memory is a bounded, coherent dialogue stream with highly correlated spans that are often duplicates. Under this shift, fixed top-k similarity retrieval tends to return redundant context, and post-hoc pruning can delete temporally linked prerequisites needed for correct reasoning. We argue retrieval should move beyond similarity matching and instead operate over latent components, following decoupling to aggregation: disentangle memories into semantic components, organise them into a hierarchy, and use this structure to drive retrieval. We propose xMemory, which builds a hierarchy of intact units and maintains a searchable yet faithful high-level node organisation via a sparsity–semantics objective that guides memory split and merge. At inference, xMemory retrieves top-down, selecting a compact, diverse set of themes and semantics for multi-fact queries, and expanding to episodes and raw messages only when it reduces the reader’s uncertainty. Experiments on LoCoMo and PerLTQA across the three latest LLMs show consistent gains in answer quality and token efficiency.

## Environment

Please check our environment document. Use

```bash
conda env create -f environment.yml
```
```base
conda activate xMemory
```

## Datasets

We open source the dataset of Locomo, it is in the evaluation/dataset, and their github is: https://github.com/snap-research/locomo. The PerltQA dataset link is: https://github.com/Elvin-Yiming-Du/PerLTQA, we use their en_v2 dataset.

## Quick Start

We open soure the code of llama model and feel free to explore it yourself if you would like to change other models. There are many engineering details you could explore to balance the inference consumption and performance, such as applying which part to detect entropy, etc. Our paper and codes here give you an initial idea of how to fit the retrieval index structure to the agent memory area. We hope for more research focusing on the retrieval index structure in the current memory area! And welcome any discussion.

Please pay attention that in our experiments we use an A100 80G GPU, and you might need to change the batch size, worker etc. to adapt to your GPUs.

# Memory Construction

Run inference with a single command. We use the add.py document in the evaluation/locomo/add.py, your deployment path is in the evaluation/.

```bash
CUDA_VISIBLE_DEVICES=0 python locomo/add.py --llm-model meta-llama/Meta-Llama-3.1-8B-Instruct --conversation-limit 1 --session-limit 3 --verbose
```

--conversation-limit and --session-limit is to fast test the memory construction, if you would like run the full dataset, you could delete them, --verbose means details the memory construction process.

Please attention: if you change to other model, or other memory path, the config.json in the evaluation/locomo need be changed and the config.py in src/config.py need be changed.

After you finished the memory constructure process, your memory path should be：
```text
evaluation_memories_llama
├── chroma_db
├── episodes
├── graph
├── semantic
├── semantic-knn
├── themes
```
And you could use gexf_view.py document to analyse the memory graph.
Or you could see our example. We provide our llama model memory of the locomo dataset at the release.

# Retrieval 
Run inference with a single command. We use the add.py document in the evaluation/locomo/xMemory_search_framework.py, your deployment path is in the evaluation/.

```bash
CUDA_VISIBLE_DEVICES=0 python locomo/xMemory_search_framework.py --llm-model meta-llama/Meta-Llama-3.1-8B-Instruct  --search-strategy adaptive_hier
```

Please attention: if you change to other model, or other memory path, the config.json in the evaluation/locomo need be changed and the config.py in src/config.py need be changed.

After you finished the memory constructure process, you will have results_github.json, results_github.llm_identity.json, results_github.token_stats.json. or you could change their name in the parameter.

# Evaluation

After retrieval, run
```bash
python locomo/evals.py
```
and
```bash
python locomo/generate_scores.py
```
and get the final result.

## Acknowledgements

This codebase is inspired by the [Nemori](https://github.com/nemori-ai/nemori) and [CAM NeurIPS 2025](https://github.com/rui9812/CAM).
.
We thank the authors for making their code publicly available, which helped us design and implement several components of xMemory.

## Citation 
If you find this repository useful, please cite our paper:

```bash
@article{hu2026beyond,
  title={Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation},
  author={Hu, Zhanghao and Zhu, Qinglin and Yan, Hanqi and He, Yulan and Gui, Lin},
  journal={arXiv preprint arXiv:2602.02007},
  year={2026}
}
```











