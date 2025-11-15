"""
utils/metrics.py

Implements core evaluation metrics:
  - BLEU, ROUGE, BERTScore (text)
  - CLIPScore, FID (image)
  - Constraint Satisfaction Rate (CSR)
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Optional heavy deps
try:
    import torch
    import clip
    _HAS_CLIP = True
except ImportError:
    _HAS_CLIP = False

try:
    from transformers import BertTokenizer, BertModel
    _HAS_BERT = True
except ImportError:
    _HAS_BERT = False

try:
    from scipy import linalg
except ImportError:
    linalg = None


# ------------------------------------------------------------
# 🧠 Text Metrics
# ------------------------------------------------------------

def compute_bleu(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute corpus-level BLEU score.
    """
    refs = [[r.split()] for r in references]
    hyps = [h.split() for h in hypotheses]
    smoothie = SmoothingFunction().method4
    return corpus_bleu(refs, hyps, smoothing_function=smoothie) * 100


def compute_rouge_l(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute average ROUGE-L F1 score across samples.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(references, hypotheses)
    ]
    return np.mean(scores) * 100


def compute_bertscore(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute average cosine similarity using BERT embeddings.
    """
    if not _HAS_BERT:
        raise ImportError("Install transformers to use BERTScore metric.")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    model.eval()
    with torch.no_grad():
        scores = []
        for ref, hyp in zip(references, hypotheses):
            inputs_ref = tokenizer(ref, return_tensors="pt", truncation=True)
            inputs_hyp = tokenizer(hyp, return_tensors="pt", truncation=True)

            emb_ref = model(**inputs_ref).last_hidden_state.mean(1)
            emb_hyp = model(**inputs_hyp).last_hidden_state.mean(1)
            sim = torch.nn.functional.cosine_similarity(emb_ref, emb_hyp)
            scores.append(sim.item())
        return np.mean(scores) * 100


# ------------------------------------------------------------
# 🎨 Image Metrics
# ------------------------------------------------------------

def compute_clipscore(images: List[Image.Image], captions: List[str]) -> float:
    """
    Compute CLIPScore (alignment between generated image and caption).
    """
    if not _HAS_CLIP:
        raise ImportError("Install openai-clip to use CLIPScore metric.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    scores = []
    for img, cap in tqdm(zip(images, captions), total=len(images), desc="CLIPScore"):
        image_input = preprocess(img).unsqueeze(0).to(device)
        text_input = clip.tokenize([cap]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).item()
            scores.append(similarity)

    return float(np.mean(scores))


def compute_fid(fake_features: np.ndarray, real_features: np.ndarray) -> float:
    """
    Compute Fréchet Inception Distance (FID) between real and generated image features.
    """
    if linalg is None:
        raise ImportError("scipy is required for FID computation.")

    mu1, sigma1 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    mu2, sigma2 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


# ------------------------------------------------------------
# ⚖️ Constraint Metrics
# ------------------------------------------------------------

def compute_constraint_satisfaction(results: List[Dict[str, Any]]) -> float:
    """
    Computes the percentage of outputs satisfying all constraints.
    Each result dict expected to have: {"factual": bool, "style": bool, "ethical": bool}
    """
    if not results:
        return 0.0

    total = len(results)
    satisfied = sum(
        1 for r in results if all(r.get(k, False) for k in ("factual", "style", "ethical"))
    )
    return (satisfied / total) * 100


# ------------------------------------------------------------
# 🧮 Unified Evaluator
# ------------------------------------------------------------

class MetricsEvaluator:
    """
    Unified interface for computing multimodal metrics.
    """

    def __init__(self, metrics: Optional[List[str]] = None):
        self.metrics = metrics or ["BLEU", "ROUGE-L", "CLIPScore", "FID", "CSR"]

    def evaluate(
        self,
        text_refs: List[str],
        text_gens: List[str],
        images: Optional[List[Image.Image]] = None,
        real_image_feats: Optional[np.ndarray] = None,
        fake_image_feats: Optional[np.ndarray] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """
        Compute all enabled metrics.
        """
        results = {}
        for metric in self.metrics:
            metric = metric.upper()
            try:
                if metric == "BLEU":
                    results["BLEU"] = compute_bleu(text_refs, text_gens)
                elif metric == "ROUGE-L":
                    results["ROUGE-L"] = compute_rouge_l(text_refs, text_gens)
                elif metric == "BERTSCORE":
                    results["BERTScore"] = compute_bertscore(text_refs, text_gens)
                elif metric == "CLIPSCORE":
                    if images is None:
                        raise ValueError("Images required for CLIPScore.")
                    results["CLIPScore"] = compute_clipscore(images, text_gens)
                elif metric == "FID":
                    if real_image_feats is None or fake_image_feats is None:
                        raise ValueError("FID requires real and fake image features.")
                    results["FID"] = compute_fid(fake_image_feats, real_image_feats)
                elif metric == "CSR":
                    if constraints is None:
                        raise ValueError("Constraint satisfaction requires constraint dicts.")
                    results["CSR"] = compute_constraint_satisfaction(constraints)
            except Exception as e:
                results[metric] = f"Error: {e}"
        return results



# ==============================================
# Content Similarity Ratio (CSR)
# ==============================================
import numpy as np

def compute_csr(predicted_texts, reference_texts):
    """
    Compute a simple Content Similarity Ratio (CSR)
    — measures semantic overlap between predicted and reference texts.
    For demonstration, we’ll compute average cosine similarity 
    using sentence-transformer embeddings if available, 
    otherwise fallback to Jaccard similarity.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("all-MiniLM-L6-v2")

        pred_emb = model.encode(predicted_texts, convert_to_tensor=True, normalize_embeddings=True)
        ref_emb = model.encode(reference_texts, convert_to_tensor=True, normalize_embeddings=True)

        cosine_scores = util.cos_sim(pred_emb, ref_emb)
        mean_score = float(np.mean(cosine_scores.cpu().numpy()))
        return mean_score

    except Exception:
        # fallback Jaccard similarity (token overlap)
        def jaccard(a, b):
            set_a, set_b = set(a.lower().split()), set(b.lower().split())
            return len(set_a & set_b) / len(set_a | set_b) if len(set_a | set_b) else 0.0

        scores = [jaccard(p, r) for p, r in zip(predicted_texts, reference_texts)]
        return float(np.mean(scores))




if __name__ == "__main__":
    # Example dry run
    refs = ["a dog playing in the park", "a cat sleeping on a couch"]
    gens = ["dog running in a park", "sleeping cat on sofa"]

    evaluator = MetricsEvaluator(metrics=["BLEU", "ROUGE-L", "CSR"])
    constraints = [{"factual": True, "style": True, "ethical": True},
                   {"factual": True, "style": False, "ethical": True}]

    result = evaluator.evaluate(
        text_refs=refs,
        text_gens=gens,
        constraints=constraints
    )
    print(result)