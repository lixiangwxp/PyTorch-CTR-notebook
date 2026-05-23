# PyTorch 推荐系统实现

这个仓库基于 PyTorch 实现了常见的推荐系统与 CTR 预测模型，适合一边看论文一边对照 notebook 跑实验，加深对模型结构和数据处理流程的理解。

- 所有模型都提供了对应的 notebook 版本，便于逐格阅读和调试。
- Embedding 相关实现思路参考了 [pytorch-fm](https://github.com/rixwew/pytorch-fm)。
- 我补充了中文注释、中文 Markdown、可直接创建的 `conda` 环境文件，以及更稳妥的数据路径解析方式。
- 新增了 Microsoft Recommenders 两个示例的 PyTorch 改写版：推荐系统评估指标，以及 MIND/NRMS 新闻推荐。
- 新增了 RecTools Transformer Models Tutorial 的 PyTorch 中文学习版：SASRec / BERT4Rec 序列推荐。

## 数据集

- `MovieLens`：`ml-latest-small` 中的 `ratings.csv`，约 100 万条记录。
- `Criteo`：截取前 100K 条样本。
- `Amazon Books`：处理后的数据来自 [DIEN-pipeline](https://github.com/kupuSs/DIEN-pipline)，截取前 100K 条样本。

## 数据处理思路

数据处理流程参考了 [Recommender-System-with-TF2.0](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0)：

- 连续特征：先分桶，再做离散化编码。
- 类别特征：直接编码成离散 ID。

## 可用模型

| 模型 | 论文 |
| --- | --- |
| 逻辑回归（LR） |  |
| 混合逻辑回归（MLR） | [Kun Gai, et al. Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction, 2017.](https://arxiv.org/abs/1704.05194) |
| GBDT + LR |  |
| 因子分解机（FM） | [S Rendle, Factorization Machines, 2010.](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) |
| 域感知因子分解机（FFM） | [Y Juan, et al. Field-aware Factorization Machines for CTR Prediction, 2015.](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) |
| Deep Crossing | [Ying Shan, et al. Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features, 2016.](http://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf) |
| 基于乘积的神经网络（PNN） | [Y Qu, et al. Product-based Neural Networks for User Response Prediction, 2016.](https://arxiv.org/abs/1611.00144) |
| Wide & Deep | [HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.](https://arxiv.org/abs/1606.07792) |
| 深度交叉网络（DCN） | [R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.](https://arxiv.org/abs/1708.05123) |
| 由 FM 支持的神经网络（FNN） | [W Zhang, et al. Deep Learning over Multi-field Categorical Data - A Case Study on User Response Prediction, 2016.](https://arxiv.org/abs/1601.02376) |
| DeepFM | [H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.](https://arxiv.org/abs/1703.04247) |
| 神经因子分解机（NFM） | [X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.](https://arxiv.org/abs/1708.05027) |
| 注意力因子分解机（AFM） | [J Xiao, et al. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks, 2017.](https://arxiv.org/abs/1708.04617) |
| 深度兴趣网络（DIN） | [Guorui Zhou, et al. Deep Interest Network for Click-Through Rate Prediction, 2017.](https://arxiv.org/abs/1706.06978) |
| 深度兴趣演化网络（DIEN） | [Guorui Zhou, et al. Deep Interest Evolution Network for Click-Through Rate Prediction, 2018.](https://arxiv.org/abs/1809.03672) |
| 隐语义因子模型（LFM） |  |
| 神经协同过滤（NeuralCF） | [X He, et al. Neural Collaborative Filtering, 2017.](https://arxiv.org/abs/1708.05031) |

## Microsoft Recommenders PyTorch 示例

- `notebook/Microsoft_Evaluation_PyTorch.ipynb`：对应 Microsoft Recommenders 的 `examples/03_evaluate/evaluation.ipynb`，用 PyTorch/Pandas 手写 RMSE、MAE、R2、Precision@K、Recall@K、NDCG@K、MAP@K、AUC、LogLoss 等指标。
- `notebook/Microsoft_NRMS_MIND_PyTorch.ipynb`：对应 Microsoft Recommenders 的 `examples/00_quick_start/nrms_MIND.ipynb`，用 PyTorch 从零实现一个可跑通的 NRMS/MIND 新闻推荐教学样例。

这两个新增 notebook 都不依赖 TensorFlow，也不依赖 Microsoft `recommenders` 包，方便直接在本仓库环境里运行和学习。

## RecTools Transformer 序列推荐 PyTorch 示例

- `rectools_transformer_models_tutorial/notebooks/01_sasrec_bert4rec_pytorch.ipynb`：对应 RecTools 官方 Transformer Models Tutorial，用纯 PyTorch 讲解 SASRec、BERT4Rec、序列 padding/mask、next-item prediction、多种 loss、Top-K 评估、item-to-item 推荐和冷启动推理。
- `rectools_transformer_models_tutorial/src/transformer_seqrec.py`：notebook 依赖的 PyTorch 实现，方便单独查看底层模型和数据处理代码。

这个子目录不依赖 TensorFlow，也不强制安装 RecTools；它把官方 tutorial 的学习重点整理成了更轻量的可运行 PyTorch 版本。

## 环境配置

仓库根目录已经提供了 `environment.yml`，推荐直接用 `conda` 创建环境：

```bash
conda env create -f environment.yml
conda activate pytorch_ctr_notebook
python -m ipykernel install --user --name pytorch_ctr_notebook --display-name "PyTorch CTR Notebook"
```

然后在仓库根目录启动 Jupyter：

```bash
jupyter lab
```

打开 `notebook/` 目录下的 `.ipynb` 文件，选择 `PyTorch CTR Notebook` 这个内核即可。

## 目录说明

- `notebook/`：适合从零阅读的实验 notebook。
- `notebook/utils/` 与 `notebook/layer/`：notebook 依赖的工具函数与基础层实现。
- `model/`：对应模型的纯 Python 版本实现。
- `dataset/`：示例数据文件。
