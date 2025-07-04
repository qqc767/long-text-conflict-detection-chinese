from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import torch
import torch.nn.functional as F
from typing import Tuple, List
import itertools
import numpy as np

class DetectConflicts:
    def __init__(self, 
                 model_path: str = "./models/DeBERTa-v3-base-mnli-fever-anli", 
                 embed_model: str = "./models/all-MiniLM-L6-v2", 
                 device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.config = AutoConfig.from_pretrained(model_path)
        self.labels = ["entailment", "neutral", "contradiction"]

        # SBERT model for embeddings
        self.embedder = SentenceTransformer(embed_model)

    def detect_pair(self, text_a: str, text_b: str, threshold, verbose) -> Tuple[str, float]:
        # a -> b
        inputs1 = self.tokenizer(text_a, text_b, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs1 = self.model(**inputs1)
        probs1 = F.softmax(outputs1.logits, dim=1)[0]

        # b -> a
        inputs2 = self.tokenizer(text_b, text_a, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs2 = self.model(**inputs2)
        probs2 = F.softmax(outputs2.logits, dim=1)[0]

        contradiction_prob = max(probs1[2], probs2[2])

        if verbose:
                print(f"[{text_a}] → [{text_b}] 推理：{self.labels[torch.argmax(probs1)]} 概率: {probs1.tolist()}")
                print(f"[{text_b}] → [{text_a}] 推理：{self.labels[torch.argmax(probs2)]} 概率: {probs2.tolist()}")
                print(f"最大冲突概率: {contradiction_prob:.4f}")

        if self.labels[torch.argmax(probs1)] == "contradiction" and self.labels[torch.argmax(probs2)] == "contradiction":
            if verbose:
                print(f"[{text_a}] → [{text_b}] 推理：{self.labels[torch.argmax(probs1)]} 概率: {probs1.tolist()}")
                print(f"[{text_b}] → [{text_a}] 推理：{self.labels[torch.argmax(probs2)]} 概率: {probs2.tolist()}")
                print(f"最大冲突概率: {contradiction_prob:.4f}")
            if contradiction_prob > threshold:
                return "冲突", float(contradiction_prob)
        return "无明显冲突", float(contradiction_prob)

    def cluster_sentences(self, sentences: List[str], threshold: float = 1.0) -> List[List[str]]:
        embeddings = self.embedder.encode(sentences)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric='euclidean', linkage='ward')
        labels = clustering.fit_predict(embeddings)
        clusters = {}
        for sent, label in zip(sentences, labels):
            clusters.setdefault(label, []).append(sent)
        return list(clusters.values())

    def detect_batch(self, sentences: List[str], threshold: float = 0.95, verbose: bool = True) -> List[Tuple[str, str, float]]:
        """
        先聚类分事件组，再组内冲突检测
        """
        conflict_results = []
        clusters = self.cluster_sentences(sentences, threshold=1.0)

        print(clusters)

        for group in clusters:
            if len(group) < 2:
                continue
            for sent1, sent2 in itertools.combinations(group, 2):
                label, prob = self.detect_pair(sent1, sent2, threshold=threshold, verbose=verbose)
                if label == "冲突":
                    conflict_results.append((sent1, sent2, prob))

        return conflict_results
