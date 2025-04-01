import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, roc_auc_score
from utils.DataProcess import CSVDataset, TokenDataset

# 数据加载
csv_file = "Data/test_data.csv"
ds = CSVDataset(csv_file)

attr_texts = [item[0] for item in ds]
desc_texts = [item[1] for item in ds]
labels = np.array([item[2] for item in ds])

# 使用TfidfVectorizer计算TF-IDF向量
vectorizer = TfidfVectorizer()
tfidf_matrix1 = vectorizer.fit_transform(attr_texts)
tfidf_matrix2 = vectorizer.transform(desc_texts)

# 计算余弦相似度
cosine_similarities = [cosine_similarity(tfidf_matrix1[i], tfidf_matrix2[i])[0][0] for i in range(len(attr_texts))]

# 将余弦相似度转换为二分类预测
threshold = 0.4
predictions = [1 if sim > threshold else 0 for sim in cosine_similarities]

# 计算F1分数
f1 = f1_score(labels, predictions)
print(f"F1 Score: {f1}")

# 计算AUC
auc = roc_auc_score(labels, predictions)
print(f"AUC: {auc}")