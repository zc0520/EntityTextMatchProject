import spacy
import requests


# 加载 spaCy 的中文模型
nlp = spacy.load("zh_core_web_sm")

# GeoNames API 配置
GEONAMES_USERNAME = "your_username"  # 替换为你的 GeoNames 用户名
GEONAMES_API_URL = "http://api.geonames.org/searchJSON"


def extract_locations(text):
    """
    从文本中提取地名
    :param text: 输入文本
    :return: 提取的地名列表
    """
    # 使用 spaCy 进行命名实体识别
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]  # GPE 表示地名
    return locations


def validate_location(location_name):
    """
    使用 GeoNames 验证地名是否存在
    :param location_name: 地名
    :return: 如果地名存在，返回 True；否则返回 False
    """
    params = {
        "q": location_name,
        "maxRows": 1,
        "username": GEONAMES_USERNAME
    }
    response = requests.get(GEONAMES_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return len(data.get("geonames", [])) > 0
    else:
        raise Exception(f"GeoNames 请求失败，状态码: {response.status_code}")


def geoparse_text(text):
    """
    从文本中提取并验证地名
    :param text: 输入文本
    :return: 提取的合法地名列表
    """
    # 提取地名
    locations = extract_locations(text)

    # 验证地名
    # valid_locations = []
    # for location in locations:
    #     if validate_location(location):
    #         valid_locations.append(location)

    return locations


# 示例使用
if __name__ == "__main__":
    text = "马山石林，山东省，青岛，即墨"
    locations = geoparse_text(text)
    print(f"提取的地名: {locations}")

