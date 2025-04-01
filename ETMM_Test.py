import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.DataProcess import CSVDataset, TokenDataset
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics import roc_auc_score
from ETMM import ETMM

batch_size = 16
# roberta_model_name = "chinese-roberta-wwm-ext"
roberta_model_name = "bert-base-chinese"
dropout = 0.5
hidden_size = 768

# 数据加载
csv_file = "Data/test_data.csv"
ds = CSVDataset(csv_file)

attr_texts = [item[0] for item in ds]
desc_texts = [item[1] for item in ds]
labels = np.array([item[2] for item in ds])

# 初始化模型和分词器
tokenizer = BertTokenizer.from_pretrained(roberta_model_name)
roberta = BertModel.from_pretrained(roberta_model_name)

# 测试数据集和 DataLoader
test_dataset = TokenDataset(attr_texts, desc_texts, labels, tokenizer)
test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

# 实例化ETMM、加载模型
model = ETMM(roberta, hidden_size, dropout)
model.load_state_dict(torch.load("model/esim_model.pth"))

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # 设置模型为评估模式

# 测试函数
def test(model, test_loader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 禁用梯度计算
        for batch in test_loader:
            input1_ids = batch["input1_ids"].to(device)
            input1_attention_mask = batch["input1_attention_mask"].to(device)
            input2_ids = batch["input2_ids"].to(device)
            input2_attention_mask = batch["input2_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 前向传播
            logits = model(input1_ids, input1_attention_mask, input2_ids, input2_attention_mask)
            probs = torch.sigmoid(logits).squeeze()  # 将 logits 转换为概率值
            preds = (probs > 0.5).int()  # 根据概率值预测类别（0 或 1）

            # 收集预测结果和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds,zero_division=1)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test AUC: {auc:.4f}")

# 运行测试
test(model, test_loader, device)

