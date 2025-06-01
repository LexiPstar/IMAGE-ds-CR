# utils/utils.py

import yaml
import re

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def clean_caption(caption):
    # 简单清洗：去标点、小写
    caption = caption.lower()
    caption = re.sub(r"[^\w\s]", "", caption)
    return caption
