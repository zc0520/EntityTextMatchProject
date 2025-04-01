import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import f1_score, roc_auc_score
from utils.DataProcess import CSVDataset

# 数据预处理
class GeoSurveyDataset(Dataset):
    def __init__(self, entities, texts, labels, word2vec_model):
        self.entities = entities  # 空间实体的文本表示
        self.texts = texts  # 外部文本描述
        self.labels = labels  # 匹配标签
        self.word2vec = word2vec_model

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        entity = self.entities[idx]
        text = self.texts[idx]
        label = self.labels[idx]

        # 将文本转换为词向量
        entity_vec = self.text_to_vector(entity)
        text_vec = self.text_to_vector(text)

        return {
            'entity': torch.tensor(entity_vec, dtype=torch.float),
            'text': torch.tensor(text_vec, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.float)
        }

    def text_to_vector(self, text):
        words = text.split()
        vectors = [self.word2vec.wv[word] for word in words if word in self.word2vec.wv]
        if not vectors:
            return np.zeros(self.word2vec.vector_size)
        return np.mean(vectors, axis=0)

class SiameseLSTMWithAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout):
        super(SiameseLSTMWithAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        # LSTM 层
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout_layer(lstm_out)

        # 注意力机制
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        return context_vector

    def similarity(self, vec1, vec2):
        # 计算两个向量的相似性
        return torch.exp(-torch.norm(vec1 - vec2, p=2, dim=1))

# 数据加载
csv_file = "Data/train_data.csv"
ds = CSVDataset(csv_file)

attr_texts = [item[0] for item in ds]
desc_texts = [item[1] for item in ds]
labels = np.array([item[2] for item in ds])


# 训练 Word2Vec 模型
sentences = [entity.split() for entity in attr_texts] + [text.split() for text in desc_texts]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 创建数据集和数据加载器
dataset = GeoSurveyDataset(attr_texts, desc_texts, labels, word2vec_model)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型参数
embedding_dim = 100
hidden_dim = 100
dropout = 0.2

model = SiameseLSTMWithAttention(embedding_dim, hidden_dim, dropout)

def train_model(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            entity = batch['entity'].unsqueeze(1)  # 添加序列维度
            text = batch['text'].unsqueeze(1)
            label = batch['label']

            optimizer.zero_grad()

            entity_vec = model(entity)
            text_vec = model(text)

            similarity = model.similarity(entity_vec, text_vec)
            loss = criterion(similarity, label)

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            entity = batch['entity'].unsqueeze(1)
            text = batch['text'].unsqueeze(1)
            label = batch['label']

            entity_vec = model(entity)
            text_vec = model(text)

            similarity = model.similarity(entity_vec, text_vec)
            preds = (similarity > 0.5).float()

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    print(f"F1 Score: {f1}, AUC: {auc}")

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# 训练模型
train_model(model, dataloader, optimizer, criterion, epochs=30)

# 数据加载
csv_file = "Data/test_data.csv"
ds = CSVDataset(csv_file)

attr_texts_test = [item[0] for item in ds]
desc_texts_test = [item[1] for item in ds]
labels = np.array([item[2] for item in ds])


# 训练 Word2Vec 模型
sentences = [entity.split() for entity in attr_texts_test] + [text.split() for text in desc_texts_test]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 创建数据集和数据加载器
dataset = GeoSurveyDataset(attr_texts_test, desc_texts_test, labels, word2vec_model)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 评估模型
evaluate_model(model, dataloader)