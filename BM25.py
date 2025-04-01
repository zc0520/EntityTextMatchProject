import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics import f1_score, roc_auc_score
from utils.DataProcess import CSVDataset, TokenDataset
import jieba

# 数据加载
csv_file = "Data/test_data.csv"
ds = CSVDataset(csv_file)

attr_texts = [item[0] for item in ds]
desc_texts = [item[1] for item in ds]
labels = np.array([item[2] for item in ds])

# 对 texts1 和 texts2 进行分词
tokenized_texts1 = [list(jieba.cut(text)) for text in attr_texts]
tokenized_texts2 = [list(jieba.cut(text)) for text in desc_texts]

# 将 tokenized_texts1 作为语料库
bm25 = BM25Okapi(tokenized_texts1)

# 计算 BM25 分数
bm25_scores = []
for query in tokenized_texts2:
    scores = bm25.get_scores(query)
    bm25_scores.append(np.max(scores))  # 取最大分数作为匹配分数

# 对BM25分数进行归一化处理，使分数区间在【0，1】
min_score = min(bm25_scores)
max_score = max(bm25_scores)
normalized_scores = [(score - min_score) / (max_score - min_score) for score in bm25_scores]

# 将 BM25 分数转换为二分类预测
threshold = 0.5
predictions = [1 if score > threshold else 0 for score in normalized_scores]

# 计算 F1 分数
f1 = f1_score(labels, predictions)
print(f"F1 Score: {f1}")

# 计算 AUC
auc = roc_auc_score(labels, predictions)
print(f"AUC: {auc}")