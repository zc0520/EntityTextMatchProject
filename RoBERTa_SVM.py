import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from utils.DataProcess import CSVDataset

# 1. 读取数据
csv_file = "Data/train_data.csv"  # 替换为你的 CSV 文件路径
ds = CSVDataset(csv_file)

attr_texts = [item[0] for item in ds]
desc_texts = [item[1] for item in ds]
labels = np.array([item[2] for item in ds])

# 2. 加载 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained("chinese-roberta-wwm-ext")
bert_model = BertModel.from_pretrained("chinese-roberta-wwm-ext")

# 3. 定义数据集类
class TextPairDataset(Dataset):
    def __init__(self, texts1, texts2, labels, tokenizer, max_length=128):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text1 = self.texts1[idx]
        text2 = self.texts2[idx]
        label = self.labels[idx]

        # 对文本对进行编码
        encoding = self.tokenizer(
            text1, text2,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# 4. 定义函数：使用 BERT 提取特征
def extract_features(texts1, texts2, tokenizer, bert_model, max_length=128, batch_size=32):
    dataset = TextPairDataset(texts1, texts2, [0] * len(texts1), tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    features = []
    bert_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # 获取 BERT 输出
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

            # 使用 [CLS] 标记的特征作为句子表示
            cls_features = last_hidden_state[:, 0, :].cpu().numpy()
            features.extend(cls_features)

    return np.array(features)

# 5. 提取 BERT 特征
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

# 提取文本对的 BERT 特征
features1 = extract_features(attr_texts, desc_texts, tokenizer, bert_model)
features2 = extract_features(desc_texts, attr_texts, tokenizer, bert_model)

# 将两个文本的特征拼接在一起
features = np.concatenate([features1, features2], axis=1)

# 6. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 7. 训练 SVM 分类器
svm_classifier = SVC(kernel="linear", probability=True)
svm_classifier.fit(X_train, y_train)

# 8. 评估模型
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# 计算 AUC-ROC
auc = roc_auc_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}, AUC: {auc:.4f},Accuracy: {accuracy:.4f}")
