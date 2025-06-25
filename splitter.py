import re

def split_sentences(text):
    # 按中文标点进行分句
    sentences = re.split(r'[。！？；\n]', text)
    return [s.strip() for s in sentences if s.strip()]
