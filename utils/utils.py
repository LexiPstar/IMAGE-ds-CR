# utils/utils.py

import yaml
import re

def load_config(path):
    """
    加载 YAML 格式的配置文件
    :param path: 配置文件的路径
    :return: 解析后的配置字典
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

def clean_caption(caption):
    """
    简单清洗文本：将文本转换为小写并去除标点符号
    :param caption: 待清洗的文本
    :return: 清洗后的文本
    """
    # 将文本转换为小写
    caption = caption.lower()
    # 使用正则表达式去除非字母数字和空格的字符
    caption = re.sub(r"[^\w\s]", "", caption)
    return caption