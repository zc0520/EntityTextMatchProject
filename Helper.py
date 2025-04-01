from transformers import BertTokenizer, BertModel



def downLoadModel():
    # 指定模型名称
    model_name = "hfl/chinese-roberta-wwm-ext"

    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 加载模型
    model = BertModel.from_pretrained(model_name)

    # 保存模型和分词器到本地
    model.save_pretrained("./chinese-roberta-wwm-ext")
    tokenizer.save_pretrained("./chinese-roberta-wwm-ext")
