from transformers import MarianMTModel, MarianTokenizer
from typing import List, Dict
import os

class ChineseToEnglishTranslator:
    def __init__(self, model_name: str = "./models/opus-mt-zh-en"):
        self.model_name = model_name
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate(self, sentences: List[str], batch_size: int = 8) -> Dict[str, str]:
        """
        将中文句子列表翻译成英文，并返回英文到中文的映射
        """
        en_cn_map = {}
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated_tokens = self.model.generate(**inputs)
            translated_texts = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

            for zh, en in zip(batch, translated_texts):
                en_cn_map[en] = zh
            # print(en_cn_map)
        return en_cn_map
    
# if __name__ == "__main__":
#     translator = ChineseToEnglishTranslator()
#     chinese_sentences = [
#         "6月21日，美军成功袭击了伊朗2处核设施，包括福尔多、纳坦兹。",
#         "福尔多是伊朗地下190米的深层核设施。",
#         "美军3架B-2轰炸机同时从密苏里州起飞，对伊朗的2处核设施进行轰炸。",
#         "6月21日，美东时间晚上10点，美军成功袭击了伊朗三个核设施，分别是福尔多、纳坦兹和伊斯法罕。",
#         "福尔多是伊朗地下90米的深层核设施，也是铀浓缩中心。",
#         "6架B-2轰炸机同时从密苏里州起飞，对伊朗的3处核设施进行轰炸。"
#     ]
#     result = translator.translate(chinese_sentences)
#     for en, zh in result.items():
#         print(f"[ZH] {zh}\n[EN] {en}\n")