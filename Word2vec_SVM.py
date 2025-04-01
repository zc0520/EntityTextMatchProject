import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils.DataProcess import CSVDataset
# 1. 读取数据
csv_file = "Data/train_data.csv"
ds = CSVDataset(csv_file)

attr_texts = [item[0] for item in ds]
desc_texts = [item[1] for item in ds]
labels = np.array([item[2] for item in ds])

# 2. 训练 Word2Vec 模型
def train_word2vec(texts, embedding_dim=100, window=5, min_count=1):
    # 将文本分词
    tokenized_texts = [text.split() for text in texts]
    # 训练 Word2Vec 模型
    word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=embedding_dim, window=window, min_count=min_count, workers=4)
    return word2vec_model

# 合并所有文本以训练 Word2Vec
all_texts = attr_texts + desc_texts
word2vec_model = train_word2vec(all_texts, embedding_dim=100)

# 保存 Word2Vec 模型
word2vec_model.save("word2vec.model")

# 3. 将文本转换为词向量序列
def text_to_sequence(texts, word2vec_model, max_seq_length):
    sequences = []
    for text in texts:
        sequence = []
        for word in text.split():
            if word in word2vec_model.wv:
                sequence.append(word2vec_model.wv[word])
        if len(sequence) < max_seq_length:
            sequence.extend([np.zeros(word2vec_model.vector_size)] * (max_seq_length - len(sequence)))
        else:
            sequence = sequence[:max_seq_length]
        sequences.append(sequence)
    return np.array(sequences)

# 参数设置
max_seq_length = 50
embedding_dim = word2vec_model.vector_size

# 将文本对转换为词向量序列
sequences1 = text_to_sequence(attr_texts, word2vec_model, max_seq_length)
sequences2 = text_to_sequence(desc_texts, word2vec_model, max_seq_length)

# 4. 提取文本特征
def extract_features(sequences):
    # 对每个文本的词向量序列取平均值，作为文本的特征
    features = np.array([np.mean(seq, axis=0) for seq in sequences])
    return features

features1 = extract_features(sequences1)  # 提取 text1 的特征
features2 = extract_features(sequences2)  # 提取 text2 的特征

# 将两个文本的特征拼接在一起
features = np.concatenate([features1, features2], axis=1)

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 6. 训练 SVM 分类器
svm_classifier = SVC(kernel="linear", probability=True)
svm_classifier.fit(X_train, y_train)

# 7. 评估模型
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# 计算 AUC-ROC
auc = roc_auc_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}, AUC: {auc:.4f},Accuracy: {accuracy:.4f}")