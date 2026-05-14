import re
import unicodedata
from dataclasses import dataclass


@dataclass
class ContaminationMatch:
    train_id: str
    holdout_id: str
    score: float
    match_type: str
    train_normalized: str
    holdout_normalized: str


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201C\u201D]", '"', text)
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _ngrams(s: str, n: int = 3) -> set[str]:
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: str, b: str, n: int = 3) -> float:
    na, nb = _ngrams(a, n), _ngrams(b, n)
    if not na and not nb:
        return 1.0
    union = na | nb
    if not union:
        return 0.0
    return len(na & nb) / len(union)


def detect_contamination(train_items, holdout_items, threshold: float):
    matches = []
    for tr in train_items:
        trn = normalize_text(tr["text"])
        for ho in holdout_items:
            hon = normalize_text(ho["text"])
            if trn == hon and trn:
                matches.append(ContaminationMatch(tr["id"], ho["id"], 1.0, "exact", trn, hon))
                continue
            score = _jaccard(trn, hon)
            if score >= threshold:
                matches.append(ContaminationMatch(tr["id"], ho["id"], round(score, 6), "fuzzy", trn, hon))
    return matches
