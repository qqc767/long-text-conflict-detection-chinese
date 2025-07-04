from splitter import split_sentences
from translator import ChineseToEnglishTranslator
from nli_detector import DetectConflicts
import sys

sys.stdout = open('conflict_detection.log', 'w', encoding='utf-8')

def read_text_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

if __name__ == "__main__":
    # 读取文件内容
    chinese_text = read_text_file('input_text_2.txt')

    print("🔹 Step 1: 中文分句")
    sentences = split_sentences(chinese_text)
    # print(sentences)

    print("🔹 Step 2: 翻译为英文")
    translator = ChineseToEnglishTranslator()
    en_cn_map = translator.translate(sentences)
    # for en, zh in en_cn_map.items():
    #     print(f"[ZH] {zh}\n[EN] {en}\n")

    print("🔹 Step 3: 进行NLI冲突检测")
    # 提取英文句子列表（传给 NLI 检测）
    en_sentences = list(en_cn_map.keys())

    # 进行英文冲突检测
    conflicts_en = DetectConflicts().detect_batch(en_sentences)

    print("\n🔍 检测到的冲突对:")
    for en1, en2, prob in conflicts_en:
        cn1 = en_cn_map.get(en1, "")
        cn2 = en_cn_map.get(en2, "")
        print(f"🟥 冲突:\n - {cn1}\n - {cn2}\n - 概率: {prob:.4f}")
