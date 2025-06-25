from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import torch.nn.functional as F
from typing import Tuple, List
import itertools

class DetectConflicts:
    def __init__(self, model_path: str = "./models/bart-large-mnli", device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.config = AutoConfig.from_pretrained(model_path)
        self.labels = ["contradiction", "neutral", "entailment"]

    def detect_pair(self, text_a: str, text_b: str, threshold: float = 0.85, verbose: bool = True) -> Tuple[str, float]:
        # a -> b
        inputs1 = self.tokenizer(text_a, text_b, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs1 = self.model(**inputs1)
        probs1 = F.softmax(outputs1.logits, dim=1)[0]

        # b -> a
        inputs2 = self.tokenizer(text_b, text_a, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs2 = self.model(**inputs2)
        probs2 = F.softmax(outputs2.logits, dim=1)[0]

        contradiction_prob = max(probs1[0], probs2[0])

        if verbose:
            print(f"[{text_a}] → [{text_b}] 推理：{self.labels[torch.argmax(probs1)]} 概率: {probs1.tolist()}")
            print(f"[{text_b}] → [{text_a}] 推理：{self.labels[torch.argmax(probs2)]} 概率: {probs2.tolist()}")
            print(f"最大冲突概率: {contradiction_prob:.4f}")

        if contradiction_prob > threshold:
            return "冲突", float(contradiction_prob)
        else:
            return "无明显冲突", float(contradiction_prob)

    def detect_batch(self, sentences: List[str], threshold: float = 0.85, verbose: bool = False) -> List[Tuple[str, str, float]]:
        """
        输入句子列表，输出所有冲突句对及概率。
        """
        conflict_results = []
        for sent1, sent2 in itertools.combinations(sentences, 2):
            label, prob = self.detect_pair(sent1, sent2, threshold=threshold, verbose=verbose)
            if label == "冲突":
                conflict_results.append((sent1, sent2, prob))
        return conflict_results

# if __name__ == "__main__":
#     detector = DetectConflicts(device="cpu")

#     sentences = [
#         "On 21 June, United States forces successfully attacked two Iranian nuclear facilities, including Fordow and Natanz.",
#         "Fordo is a deep-sea nuclear facility 190 metres underground in Iran.",
#         "Three United States B-2 bombers took off simultaneously from Missouri State and bombed two Iranian nuclear facilities.",
#         "On 21 June, at 10 p.m. U.S. Eastern Time, United States forces successfully attacked three Iranian nuclear facilities: Fordo, Natanz and Isfahan.",
#         "Fordow is a deep-sea nuclear facility 90 metres underground in Iran and a uranium enrichment centre.",
#         "Six B-2 bombers also took off from Missouri State and bombed three Iranian nuclear facilities."
#     ]

#     results = detector.detect_batch(sentences, threshold=0.85, verbose=True)

#     print("\n=== 冲突句对检测结果 ===")
#     for s1, s2, prob in results:
#         print(f"[S1] {s1}\n[S2] {s2}\n[冲突概率] {prob:.4f}\n")