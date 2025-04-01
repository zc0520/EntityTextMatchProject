import pandas as pd
import random
import torch
from torch.utils.data import Dataset
import jieba

# 自定义数据集
class TokenDataset(Dataset):
    def __init__(self, texts1, texts2, labels, tokenizer, max_len=128):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, idx):
        text1 = self.texts1[idx]
        text2 = self.texts2[idx]
        label = self.labels[idx]

        encoding1 = self.tokenizer(text1, return_tensors="pt", max_length=self.max_len, padding="max_length", truncation=True)
        encoding2 = self.tokenizer(text2, return_tensors="pt", max_length=self.max_len, padding="max_length", truncation=True)

        return {
            "input1_ids": encoding1["input_ids"].flatten(),
            "input1_attention_mask": encoding1["attention_mask"].flatten(),
            "input2_ids": encoding2["input_ids"].flatten(),
            "input2_attention_mask": encoding2["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.float)
        }

# 自定义Dataset类,读取CSV文件
class CSVDataset(Dataset):
    def __init__(self, csv_file):
        # 读取CSV文件
        self.data = pd.read_csv(csv_file, encoding='ANSI')

    def __len__(self):
        # 返回数据集的总行数
        return len(self.data)

    def __getitem__(self, idx):
        # 获取当前行的数据
        row = self.data.iloc[idx]

        # 拼接除“描述文本”和“标签”外的其他字段
        other_fields = row.iloc[:-2]  # 获取除最后两列外的所有列
        attr_fields = [str(field) for idx, field in enumerate(other_fields) if idx not in {0, 2, 3}]
        attr_texts = ' '.join(attr_fields)  # 拼接为文本

        # attr_texts =  ' '.join([str(field) for field in other_fields])

        # 获取描述文本和标签
        desc_texts = row.iloc[-2]  # 倒数第二列是描述文本
        labels = row.iloc[-1]       # 最后一列是标签

        # 返回拼接文本、描述文本和标签
        return attr_texts, desc_texts, labels


class TextPairDataset(Dataset):
    def __init__(self, texts1, texts2, labels, vocab, max_len=50):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text1 = self.texts1[idx]
        text2 = self.texts2[idx]
        label = self.labels[idx]

        # 分词并转换为索引
        text1_idx = [self.vocab.get(word, self.vocab['<unk>']) for word in jieba.cut(text1)][:self.max_len]
        text2_idx = [self.vocab.get(word, self.vocab['<unk>']) for word in jieba.cut(text2)][:self.max_len]

        # 获取实际长度
        len1 = len(text1_idx)
        len2 = len(text2_idx)

        # 填充到固定长度
        text1_idx = self.pad_sequence(text1_idx, self.max_len)
        text2_idx = self.pad_sequence(text2_idx, self.max_len)

        return (
            torch.tensor(text1_idx, dtype=torch.long),
            torch.tensor(text2_idx, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
            len1,  # 返回 text1 的实际长度
            len2   # 返回 text2 的实际长度
        )

    def pad_sequence(self, sequence, max_len):
        if len(sequence) < max_len:
            sequence = sequence + [self.vocab['<pad>']] * (max_len - len(sequence))
        else:
            sequence = sequence[:max_len]
        return sequence

# 创建负样本，每个正样本生成四个负样本
def createNegativesample():
    # 读取 CSV 文件
    file_path = 'Data/data+.csv'  # 替换为你的 CSV 文件路径
    df = pd.read_csv(file_path, encoding='ANSI')

    # 获取所有描述文本的列表
    descriptions = df["描述文本"].tolist()

    # 生成负样本数据
    negative_samples = []
    for idx in range(len(df)):
        for i in range(1):
            # 随机选择一条描述文本，但不能是当前记录的描述文本
            description = random.choice([d for d in descriptions if d != df.loc[idx, "描述文本"]])
            negative_sample = df.iloc[idx].copy()
            negative_sample["描述文本"] = description
            negative_sample["标签"] = 0
            negative_samples.append(negative_sample)

    # 将负样本数据转换为 DataFrame
    negative_df = pd.DataFrame(negative_samples)

    # 如果需要保存为 CSV 文件
    negative_df.to_csv("negative_samples.csv", index=False)


# 主程序
if __name__ == "__main__":
    createNegativesample();