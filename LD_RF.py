import numpy as np
import Levenshtein
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from utils.DataProcess import CSVDataset, TokenDataset

# 数据加载
csv_file = "Data/train_data.csv"
ds = CSVDataset(csv_file)

attr_texts = [item[0] for item in ds]
desc_texts = [item[1] for item in ds]
labels = np.array([item[2] for item in ds])

# 计算编辑距离
levenshtein_distances = [Levenshtein.distance(text1, text2) for text1, text2 in zip(attr_texts, desc_texts)]

# 构建特征向量
features = []
for text1, text2, dist in zip(attr_texts, desc_texts, levenshtein_distances):
    length1 = len(text1)
    length2 = len(text2)
    length_diff = abs(length1 - length2)
    features.append([dist, length1, length2, length_diff])

# 转换为 NumPy 数组
X = np.array(features)
y = np.array(labels)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf_classifier.fit(X_train, y_train)

# 预测概率（用于 AUC）
y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]

# 预测类别（用于 F1）
y_pred = rf_classifier.predict(X_test)

# 计算 F1 分数
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1}")

# 计算 AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc}")