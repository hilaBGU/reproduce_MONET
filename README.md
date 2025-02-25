# Enhanced-MONET: Improving Modality-Embracing Graph Convolutional Network and Target-Aware Attention for Multimedia Recommendation

This repository provides an implementation of *MONET* based on the original code, but with **enhancements** that improve its performance by addressing some of its limitations. Our modifications include:

- **Learnable Parameters**: We introduce dynamic learning for modality weighting (`α, β`) instead of using predefined values.
- **Enhanced Modality Preservation**: We incorporate weighted fusion of initial and learned embeddings to prevent loss of modality-specific information.
- **User-Aware Target-Oriented Attention**: We refine the attention mechanism by integrating user representations to improve personalization.

These modifications improve the adaptability and effectiveness of MONET for multimedia recommendation.

## Reference Paper

If you are interested in our modifications, please refer to the following paper:

> **Enhanced-MONET: Addressing Modality Fusion and User Adaptation in Multimedia Recommendation**  
> *Beni Ifland, Bar Lazar Dolev, Hila Zylfi*  
> February 2025

This work builds upon:

> MONET: Modality-Embracing Graph Convolutional Network and Target-Aware Attention for Multimedia Recommendation  
> *Yungi Kim, Taeri Kim, Won-Yong Shin, and Sang-Wook Kim*  
> 17th ACM International Conference on Web Search and Data Mining (ACM WSDM 2024)

---

## Overview of MONET and Our Enhancements

MONET improves multimedia recommendation by leveraging **modality-embracing graph convolutional networks** and **target-aware attention mechanisms**. However, MONET has limitations, including:
- **Fixed modality weighting**: The predefined values for modality importance may not generalize across datasets.
- **Modality information loss**: GCN-learned embeddings replace original embeddings, discarding valuable information.
- **Limited personalization**: MONET's target-aware attention does not consider user-specific adaptations.

To overcome these challenges, **Enhanced-MONET** introduces:
1. **Dynamic Learnable Parameters (`α, β`)** – Allowing the model to optimize these values instead of using static sensitivity analysis results.
2. **Improved Modality Preservation (`γ, δ`)** – Ensuring a balance between initial and learned embeddings, preventing loss of important modality signals.
3. **User-Aware Target-Oriented Attention (`ω`)** – Integrating user embeddings into the attention mechanism to enhance personalization.

---

## Authors

### Enhanced-MONET Contributors:
- **Beni Ifland**  ([ifliandb@post.bgu.ac.il](mailto:ifliandb@post.bgu.ac.il))
- **Bar Lazar Dolev** ([dobar@post.bgu.ac.il](mailto:dobar@post.bgu.ac.il))
- **Hila Zylfi** ([hilakese@post.bgu.ac.il](mailto:hilakese@post.bgu.ac.il))

### Authors of the Original MONET Paper:
- **Yungi Kim** ([gozj3319@hanyang.ac.kr](mailto:gozj3319@hanyang.ac.kr))
- **Taeri Kim** ([taerik@hanyang.ac.kr](mailto:taerik@hanyang.ac.kr))
- **Won-Yong Shin** ([wy.shin@yonsei.ac.kr](mailto:wy.shin@yonsei.ac.kr))
- **Sang-Wook Kim** ([wook@hanyang.ac.kr](mailto:wook@hanyang.ac.kr))

### Environment Setup

To successfully reproduce the results, we used the following environment setup as some package versions differ from the original implementation:

```bash
conda create -n reproduce_monet python=3.8
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install gensim==3.8.3
pip install sentence-transformers==2.2.0
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.1+cu113.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.10.1+cu113.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.10.1+cu113.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.10.1+cu113.html
python -m pip install torch_geometric==2.2.0
pip install huggingface_hub==0.8.1
pip install transformers==4.21.0
```

### Dataset Preparation

#### Dataset Download

- **Men Clothing and Women Clothing**: Download from the Amazon product dataset provided by [MAML](https://github.com/liufancs/MAML). Place the data folder in `data/`.

#### Dataset Preprocessing

Run the following command:

```bash
python build_data.py --name={Dataset}
```

### Usage

We conducted experiments specifically for **RQ1**.

#### Example: Women Clothing Dataset

- **For MONET in RQ1:**

```bash
python main.py --agg=concat --n_layers=2 --alpha=1.0 --beta=0.3 --dataset=WomenClothing --model_name=MONET_2_10_3
```

## Results Comparison

We evaluated our modifications on the **MenClothing** and **WomenClothing** datasets.

### **WomenClothing Dataset**
| Model             | Precision@20 | Recall@20 | NDCG@20 |
|------------------|------------|----------|---------|
| Original MONET  | 0.0050      | 0.0990   | 0.0450  |
| **Enhanced-MONET** | **0.0057**  | **0.1034** | **0.0468** |

### **MenClothing Dataset**
| Model             | Precision@20 | Recall@20 | NDCG@20 |
|------------------|------------|----------|---------|
| Original MONET  | 0.0045      | 0.0895   | 0.0406  |
| **Enhanced-MONET** | **0.0052**  | **0.0948** | **0.0419** |

Our modifications demonstrate a **consistent improvement** across all key evaluation metrics.  
- **Higher Recall@20** suggests that Enhanced-MONET is retrieving more relevant recommendations.  
- **Higher NDCG@20** indicates better ranking quality of recommended items.  
- **Improvements in Precision@20** show that more top-ranked items are relevant to users.

These results confirm that dynamically learning modality weights and incorporating **user-aware attention mechanisms** lead to better recommendation performance.

### Cite the Original Work

If you use this code, please cite the original paper:

```bibtex
@inproceedings{kim24wsdm,
  author   = {Yungi Kim and Taeri Kim and Won{-}Yong Shin and Sang{-}Wook Kim},
  title    = {MONET: Modality-Embracing Graph Convolutional Network and Target-Aware Attention for Multimedia Recommendation},
  booktitle = {ACM International Conference on Web Search and Data Mining (ACM WSDM 2024)},
  year     = {2024}
}
```

### Acknowledgement

This project is based on the original MONET codebase, with structural influences from [LATTICE](https://github.com/CRIPAC-DIG/LATTICE). We thank the authors for their contributions.

