import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import jieba
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from transformers import BertTokenizer, BertModel
from utils.DataProcess import CSVDataset

class TextPairDataset(Dataset):
    def __init__(self, texts1, texts2, labels, tokenizer, max_len=128):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text1 = self.texts1[idx]
        text2 = self.texts2[idx]
        label = self.labels[idx]

        # 使用 BERT Tokenizer 对文本进行编码
        encoding1 = self.tokenizer.encode_plus(
            text1,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        encoding2 = self.tokenizer.encode_plus(
            text2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids1': encoding1['input_ids'].squeeze(0),
            'attention_mask1': encoding1['attention_mask'].squeeze(0),
            'input_ids2': encoding2['input_ids'].squeeze(0),
            'attention_mask2': encoding2['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

class GeoER(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', hidden_dim=256, dropout=0.5):
        super(GeoER, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size + hidden_dim, 1)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, distance_embedding):
        # 获取 BERT 的输出
        outputs1 = self.bert(input_ids=input_ids1, attention_mask=attention_mask1)
        outputs2 = self.bert(input_ids=input_ids2, attention_mask=attention_mask2)

        # 取 [CLS] 位置的输出
        cls_output1 = outputs1.last_hidden_state[:, 0, :]
        cls_output2 = outputs2.last_hidden_state[:, 0, :]

        # 计算文本相似度
        text_similarity = torch.exp(-torch.norm(cls_output1 - cls_output2, dim=1))

        # 将文本相似度和距离嵌入拼接
        combined = torch.cat([text_similarity.unsqueeze(1), distance_embedding], dim=1)

        # 通过全连接层输出最终结果
        output = self.fc(self.dropout(combined))
        return output.squeeze(1)

# 数据加载
csv_file = "Data/train_data.csv"
ds = CSVDataset(csv_file)

attr_texts = [item[0] for item in ds]
desc_texts = [item[1] for item in ds]
labels = np.array([item[2] for item in ds])

# 加载 BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


# 划分训练集和测试集
train_texts1, test_texts1, train_texts2, test_texts2, train_labels, test_labels = train_test_split(attr_texts, desc_texts, labels, test_size=0.2, random_state=42)

# 创建数据集和数据加载器
train_dataset = TextPairDataset(train_texts1, train_texts2, train_labels, tokenizer)
test_dataset = TextPairDataset(test_texts1, test_texts2, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 初始化模型
model = GeoER()

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        optimizer.zero_grad()
        input_ids1 = batch['input_ids1']
        attention_mask1 = batch['attention_mask1']
        input_ids2 = batch['input_ids2']
        attention_mask2 = batch['attention_mask2']
        labels = batch['label']

        # 假设距离嵌入是随机生成的（实际应用中应根据实际距离计算）
        distance_embedding = torch.randn(input_ids1.size(0), 256)

        # 前向传播
        predictions = model(input_ids1, attention_mask1, input_ids2, attention_mask2, distance_embedding)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in iterator:
            input_ids1 = batch['input_ids1']
            attention_mask1 = batch['attention_mask1']
            input_ids2 = batch['input_ids2']
            attention_mask2 = batch['attention_mask2']
            labels = batch['label']

            # 假设距离嵌入是随机生成的（实际应用中应根据实际距离计算）
            distance_embedding = torch.randn(input_ids1.size(0), 256)

            # 前向传播
            predictions = model(input_ids1, attention_mask1, input_ids2, attention_mask2, distance_embedding)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            all_predictions.extend(torch.sigmoid(predictions).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return epoch_loss / len(iterator), all_predictions, all_labels


N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.3f}')

test_loss, predictions, labels = evaluate(model, test_loader, criterion)

# 计算 F1 和 AUC
predictions = np.round(predictions)
f1 = f1_score(labels, predictions)
auc = roc_auc_score(labels, predictions)
print(f'Test Loss: {test_loss:.3f}, F1 Score: {f1:.3f}, AUC: {auc:.3f}')