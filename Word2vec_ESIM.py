import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from utils.DataProcess import CSVDataset
from sklearn.metrics import roc_auc_score

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

# 4. 划分训练集和测试集
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(sequences1, sequences2, labels, test_size=0.2, random_state=42)

# 将数据转换为 PyTorch 张量
X1_train = torch.tensor(X1_train, dtype=torch.float32)
X2_train = torch.tensor(X2_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X1_test = torch.tensor(X1_test, dtype=torch.float32)
X2_test = torch.tensor(X2_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 5. 创建 DataLoader
class TextPairDataset(Dataset):
    def __init__(self, X1, X2, y):
        self.X1 = X1
        self.X2 = X2
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]

train_dataset = TextPairDataset(X1_train, X2_train, y_train)
test_dataset = TextPairDataset(X1_test, X2_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 6. 构建 ESIM 模型
# 6. 构建 ESIM 模型
class ESIM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_rate=0.5):
        super(ESIM, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # 双向 LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # 双向 LSTM 编码
        lstm1, _ = self.lstm(x1)  # [batch_size, max_seq_length, hidden_dim * 2]
        lstm2, _ = self.lstm(x2)  # [batch_size, max_seq_length, hidden_dim * 2]

        # 注意力机制
        attention = torch.bmm(lstm1, lstm2.transpose(1, 2))  # [batch_size, max_seq_length, max_seq_length]
        attention1 = torch.softmax(attention, dim=2)  # [batch_size, max_seq_length, max_seq_length]
        attention2 = torch.softmax(attention, dim=1)  # [batch_size, max_seq_length, max_seq_length]

        # 加权求和
        weighted1 = torch.bmm(attention1, lstm2)  # [batch_size, max_seq_length, hidden_dim * 2]
        weighted2 = torch.bmm(attention2.transpose(1, 2), lstm1)  # [batch_size, max_seq_length, hidden_dim * 2]

        # 拼接
        merged1 = torch.cat([lstm1, weighted1], dim=2)  # [batch_size, max_seq_length, hidden_dim * 4]
        merged2 = torch.cat([lstm2, weighted2], dim=2)  # [batch_size, max_seq_length, hidden_dim * 4]

        # 池化操作
        pooled1 = merged1.mean(dim=1)  # [batch_size, hidden_dim * 4]
        pooled2 = merged2.mean(dim=1)  # [batch_size, hidden_dim * 4]

        # 拼接并输出
        merged = torch.cat([pooled1, pooled2], dim=1)  # [batch_size, hidden_dim * 8]
        output = self.fc(merged)  # [batch_size, 1]
        return output

# 初始化模型
model = ESIM(embedding_dim=embedding_dim, hidden_dim=300)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# 7. 训练模型
def train(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X1, X2, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X1, X2)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

train(model, train_loader, optimizer, criterion, epochs=30)

# 8. 评估模型
def evaluate(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X1, X2, y in test_loader:
            outputs = model(X1, X2)
            y_true.extend(y.tolist())
            y_pred.extend((outputs > 0.5).float().tolist())
    f1 = f1_score(y_true, y_pred)
    # 计算 AUC-ROC
    auc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"F1 Score: {f1:.4f}, AUC: {auc:.4f},Accuracy: {accuracy:.4f}")

evaluate(model, test_loader)