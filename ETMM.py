import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from utils.DataProcess import CSVDataset, TokenDataset
from utils.ESIM import ESIM
import pandas as pd
from sklearn.metrics import roc_auc_score

class ETMM(nn.Module):
    def __init__(self, roberta, hidden_size, dropout=0.5):
        super(ETMM, self).__init__()
        self.roberta = roberta
        self.esim = ESIM(hidden_size, dropout)

    def forward(self, input1_ids, input1_attention_mask, input2_ids, input2_attention_mask):
        # Extract BERT embeddings
        input1_embeddings = self.roberta(input_ids=input1_ids, attention_mask=input1_attention_mask).last_hidden_state
        input2_embeddings = self.roberta(input_ids=input2_ids, attention_mask=input2_attention_mask).last_hidden_state

        # Pass embeddings to ESIM model
        logits = self.esim(input1_embeddings, input2_embeddings)
        return logits

# 训练函数
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input1_ids = batch["input1_ids"].to(device)
        input1_attention_mask = batch["input1_attention_mask"].to(device)
        input2_ids = batch["input2_ids"].to(device)
        input2_attention_mask = batch["input2_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 将标签从 0/1 转换为 -1/1
        # labels = 2 * labels - 1

        optimizer.zero_grad()
        logits = model(input1_ids, input1_attention_mask, input2_ids, input2_attention_mask)

        # 计算 Hinge Loss
        # loss = F.hinge_embedding_loss(logits.squeeze(), labels)

        loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Train Loss: {total_loss / len(dataloader)}")
    return total_loss / len(dataloader)


# 评估函数
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input1_ids = batch["input1_ids"].to(device)
            input1_attention_mask = batch["input1_attention_mask"].to(device)
            input2_ids = batch["input2_ids"].to(device)
            input2_attention_mask = batch["input2_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input1_ids, input1_attention_mask, input2_ids, input2_attention_mask)
            preds = torch.sigmoid(logits).squeeze().cpu().numpy()

            # all_preds.extend(preds > 0.5)
            all_preds.extend(np.where(preds >= 0.5, 1, 0))
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=1)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    # 计算 AUC-ROC
    auc = roc_auc_score(all_labels, all_preds)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
    return accuracy, precision, recall, f1, auc

# 主程序
if __name__ == "__main__":
    batch_size = 16
    epoch = 30
    dropout = 0.5
    hidden_size = 768
    learn_rate = 1e-5;
    # roberta_model_name = "chinese-roberta-wwm-ext"
    roberta_model_name = "bert-base-chinese"
     # 数据加载
    csv_file = "Data/train_data.csv"
    ds = CSVDataset(csv_file)

    attr_texts = [item[0] for item in ds]
    desc_texts = [item[1] for item in ds]
    labels = np.array([item[2] for item in ds])

    # 数据划分 80%的训练数据集，20%的验证数据集，测试数据集已经筛选出来了
    train_texts1, val_texts1, train_texts2, val_texts2, train_labels, val_labels = train_test_split(
            attr_texts, desc_texts, labels, test_size=0.2, random_state=42)

    # 记录训练开始时间
    start_time = time.time()

    # 初始化模型和分词器
    tokenizer = BertTokenizer.from_pretrained(roberta_model_name)
    roberta = BertModel.from_pretrained(roberta_model_name)
    # 实例化ETMM
    model = ETMM(roberta, hidden_size, dropout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

     # 创建数据集和数据加载器
    train_dataset = TokenDataset(train_texts1, train_texts2, train_labels, tokenizer)
    val_dataset = TokenDataset(val_texts1, val_texts2, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    # 训练和评估
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    results = []
    for epoch_id in range(epoch):
        print(f"Epoch {epoch_id + 1}")
        loss = train(model, train_loader, optimizer, device)
        eval_res = evaluate(model, val_loader, device)
        # 结果保存
        res = {}
        res["Epoch"] = epoch_id + 1
        res["Loss"] = loss
        res["Accuracy"] = eval_res[0]
        res["Precision"] = eval_res[1]
        res["Recall"] = eval_res[2]
        res["F1 Score"] = eval_res[3]
        res["AUC"] = eval_res[4]
        results.append(res)

    # 将评价结果数据转换为 DataFrame，并保存为csv文件
    result_df = pd.DataFrame(results)
    result_df.to_csv("result.csv", index=False)

    # 保存模型权重
    model_path = "model/esim_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 记录训练结束时间
    end_time = time.time()
    # 计算训练时间
    training_time = end_time - start_time
    print(f'Training completed in {training_time:.2f} seconds')


