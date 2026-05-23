# RecTools Transformer Models Tutorial 的 PyTorch 中文学习版

这个文件夹对应 RecTools 官方文档里的 **RecSys Transformer Models Tutorial**，主题是 SASRec 和 BERT4Rec。原 tutorial 重点解释如何使用 RecTools 的 Transformer 模型，以及这些模型底层如何工作；这里把学习材料整理成纯 PyTorch、中文注释、离线可跑通的 notebook。

官方参考：

- RecTools Transformer Models Tutorial: https://rectools.readthedocs.io/en/stable/examples/tutorials/transformers_tutorial.html
- RecTools GitHub: https://github.com/MobileTeleSystems/RecTools

## 文件说明

- `notebooks/01_sasrec_bert4rec_pytorch.ipynb`：主 notebook，覆盖 SASRec、BERT4Rec、序列数据组织、padding/mask、next-item prediction、多种 loss、Top-K 评估、item-to-item 推荐和冷启动推理。
- `src/transformer_seqrec.py`：notebook 使用的 PyTorch 实现，包含数据集、Transformer 模型、loss、评估函数和推荐函数。
- `data/`：预留给你后续放真实序列推荐数据，目前 notebook 使用内置 toy 数据，保证离线可跑。

## 运行方式

复用仓库根目录的 conda 环境：

```bash
conda activate pytorch_ctr_notebook
jupyter lab
```

然后打开：

```text
rectools_transformer_models_tutorial/notebooks/01_sasrec_bert4rec_pytorch.ipynb
```

选择 `PyTorch CTR Notebook` 内核运行即可。

## 学习重点

- SASRec：因果 attention，只看当前位置之前的历史，用 shifted sequence 做下一个物品预测。
- BERT4Rec：双向 attention，随机 mask 序列中的 item，用 masked item prediction 训练。
- 序列推荐数据组织：按用户和时间排序，保留最近行为，左侧 padding。
- mask 机制：`PAD` 不参与 attention；BERT4Rec 额外使用 `MASK` token 做训练目标。
- loss 选择：完整 softmax、sampled BCE、BPR。
- Top-K 评估：HitRate@K、NDCG@K、MRR@K。
- 推理扩展：item-to-item 推荐、冷启动用户推理。
