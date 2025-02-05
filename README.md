# Reproduc-MONET: Reproducing Results for Modality-Embracing Graph Convolutional Network and Target-Aware Attention for Multimedia Recommendation

This repository provides an implementation of *MONET* based on the original code, aiming to reproduce the results presented in the following paper:

> MONET: Modality-Embracing Graph Convolutional Network and Target-Aware Attention for Multimedia Recommendation\
> Yungi Kim, Taeri Kim, Won-Yong Shin, and Sang-Wook Kim\
> 17th ACM International Conference on Web Search and Data Mining (ACM WSDM 2024)

### Overview of MONET

MONET is designed to improve multimedia recommendation by leveraging modality-embracing graph convolutional networks and target-aware attention mechanisms.

### Authors of the Original Paper

- Yungi Kim ([gozj3319@hanyang.ac.kr](mailto\:gozj3319@hanyang.ac.kr))
- Taeri Kim ([taerik@hanyang.ac.kr](mailto\:taerik@hanyang.ac.kr))
- Won-Yong Shin ([wy.shin@yonsei.ac.kr](mailto\:wy.shin@yonsei.ac.kr))
- Sang-Wook Kim ([wook@hanyang.ac.kr](mailto\:wook@hanyang.ac.kr))


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

### Reproduced Results

We conducted experiments for RQ1 on the Women Clothing and Men Clothing datasets. The reproduced results are as follows:

#### Women Clothing

- **Precision\@20:** 0.00666 (original: 0.0050)
- **Recall\@20:** 0.0962 (original: 0.0990)
- **NDCG\@20:** 0.0439 (original: 0.0450)

#### Men Clothing

- **Precision\@20:** 0.00646 (original: 0.0045)
- **Recall\@20:** 0.0927 (original: 0.0895)
- **NDCG\@20:** 0.0403 (original: 0.0406)

While our results are close to the original publication, minor differences may be due to variations in preprocessing, random seeds, or hardware environments.

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

