


import subprocess
import sys
import os
import shutil

subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "torch", "transformers", "wordfreq", "scikit-learn", "numpy", "pandas",
     "--break-system-packages", "--quiet"],
    check=True,
)

import re
import math
import string
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from wordfreq import zipf_frequency
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity






_GPT2_TOKENIZER = None
_GPT2_MODEL = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_gpt2():
    global _GPT2_TOKENIZER, _GPT2_MODEL
    if _GPT2_MODEL is None:
        _GPT2_TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
        _GPT2_MODEL = GPT2LMHeadModel.from_pretrained("gpt2").to(_DEVICE)
        _GPT2_MODEL.eval()
    return _GPT2_TOKENIZER, _GPT2_MODEL


@torch.no_grad()
def _gpt2_token_logprobs(text):
    if not text.strip():
        return []
    tokenizer, model = _load_gpt2()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc["input_ids"].to(_DEVICE)
    if input_ids.shape[1] < 2:
        return []
    outputs = model(input_ids)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return token_logprobs[0].cpu().tolist()


def _gpt2_perplexity(text):
    lps = _gpt2_token_logprobs(text)
    if not lps:
        return 0.0
    avg_neg_logprob = -_mean(lps)
    return float(math.exp(min(avg_neg_logprob, 20)))


STOPWORDS = set("""
a an the this that these those is am are was were be been being
i you he she it we they me him her us them my your his its our their
and or but if then so because as while of to in on at for with from by
about into through during before after above below over under again
further not no nor only own same too very s t can will just don should
now do does did doing have has had having
""".split())

CONJUNCTIONS = set("""
and or but nor yet so although though because since unless while
whereas if when after before until as than that
""".split())

VERB_SUFFIXES = ("ing", "ed", "ate", "ise", "ize", "fy")
NOUN_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ance", "ence", "er", "or", "ism")
ADJ_SUFFIXES = ("able", "ible", "al", "ful", "ous", "ive", "ic", "less")
ADV_SUFFIXES = ("ly",)

COMMON_VERBS = set("""
is are was were be been being do does did have has had make made go went
get got take took see saw know knew think thought say said come came
use used find found give gave tell told ask asked show showed try tried
call called need needed feel felt seem seemed leave left put set become
becomes provide provides provides allow allows helps help helped create
created require requires
""".split())

PASSIVE_AUX = ("is", "are", "was", "were", "been", "being", "be")

SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])|(?<=[.!?])\s*$')
WORD_RE = re.compile(r"[A-Za-z']+")


def _safe_div(a, b):
    return a / b if b else 0.0


def _split_sentences(text):
    text = text.strip()
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _words(text):
    return WORD_RE.findall(text)


def _std(values):
    return float(np.std(values)) if len(values) > 1 else 0.0


def _mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def _yules_k(word_list):
    n = len(word_list)
    if n == 0:
        return 0.0
    freqs = Counter(w.lower() for w in word_list)
    m1 = n
    m2 = sum(f ** 2 for f in freqs.values())
    if m1 == 0:
        return 0.0
    return 10000 * (m2 - m1) / (m1 ** 2)


def _mattr(word_list, window=20):
    n = len(word_list)
    if n == 0:
        return 0.0
    if n <= window:
        uniq = len(set(w.lower() for w in word_list))
        return _safe_div(uniq, n)
    ratios = []
    for i in range(n - window + 1):
        chunk = [w.lower() for w in word_list[i:i + window]]
        ratios.append(_safe_div(len(set(chunk)), window))
    return _mean(ratios)


def _flesch_reading_ease(word_list, sentence_list):
    def count_syllables(word):
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    n_words = len(word_list)
    n_sents = max(len(sentence_list), 1)
    if n_words == 0:
        return 0.0
    n_syll = sum(count_syllables(w) for w in word_list)
    score = 206.835 - 1.015 * (n_words / n_sents) - 84.6 * (n_syll / n_words)
    return float(score)


def _pos_ratios(word_list):
    n = len(word_list)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    verbs = nouns = adjs = advs = 0
    for w in word_list:
        wl = w.lower()
        if wl in COMMON_VERBS or wl.endswith(VERB_SUFFIXES):
            verbs += 1
        elif wl.endswith(ADV_SUFFIXES):
            advs += 1
        elif wl.endswith(ADJ_SUFFIXES):
            adjs += 1
        elif wl.endswith(NOUN_SUFFIXES) or (w[0:1].isupper() and wl not in STOPWORDS):
            nouns += 1
    return (
        _safe_div(verbs, n),
        _safe_div(nouns, n),
        _safe_div(adjs, n),
        _safe_div(advs, n),
    )


def _passive_ratio(sentences):
    if not sentences:
        return 0.0
    passive = 0
    for s in sentences:
        toks = [w.lower() for w in _words(s)]
        for i, t in enumerate(toks):
            if t in PASSIVE_AUX:
                nxt = toks[i + 1:i + 3]
                if any(n.endswith("ed") or n.endswith("en") for n in nxt):
                    passive += 1
                    break
    return _safe_div(passive, len(sentences))


def _ngram_repetition_ratio(words_lower, n):
    if len(words_lower) < n:
        return 0.0
    grams = [tuple(words_lower[i:i + n]) for i in range(len(words_lower) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    return _safe_div(repeated, len(grams))


def _top_ngram_ratio(words_lower, n=3):
    if len(words_lower) < n:
        return 0.0
    grams = [tuple(words_lower[i:i + n]) for i in range(len(words_lower) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    top = counts.most_common(1)[0][1]
    return _safe_div(top, len(grams))


def _sentence_similarity_features(sentences):
    if len(sentences) < 2:
        return 0.0, 0.0
    try:
        vec = CountVectorizer().fit_transform(sentences)
        sims = []
        arr = vec.toarray()
        for i in range(len(sentences) - 1):
            a, b = arr[i:i + 1], arr[i + 1:i + 2]
            if a.sum() == 0 or b.sum() == 0:
                sims.append(0.0)
            else:
                sims.append(float(cosine_similarity(a, b)[0, 0]))
        return _mean(sims), _std(sims)
    except ValueError:
        return 0.0, 0.0


def _structural_parallelism_ratio(sentences):
    if len(sentences) < 2:
        return 0.0
    sigs = []
    for s in sentences:
        w = _words(s)
        if not w:
            continue
        first = w[0].lower()
        length_bucket = len(w) // 5
        sigs.append((first, length_bucket))
    if not sigs:
        return 0.0
    counts = Counter(sigs)
    repeated = sum(c for c in counts.values() if c > 1)
    return _safe_div(repeated, len(sigs))


def features(text):
    text = text if isinstance(text, str) else str(text)
    sentences = _split_sentences(text)
    words = _words(text)
    words_lower = [w.lower() for w in words]
    n_words = len(words)

    word_lens = [len(w) for w in words]
    sent_word_counts = [len(_words(s)) for s in sentences] or [0]
    sent_char_lens = [len(s) for s in sentences] or [0]

    doc_logprobs = _gpt2_token_logprobs(text)
    sent_perplexities = []
    for s in sentences:
        lps = _gpt2_token_logprobs(s)
        if lps:
            sent_perplexities.append(float(math.exp(min(-_mean(lps), 20))))

    f_perplexity = _gpt2_perplexity(text)
    f_perplexity_variance = _std(sent_perplexities)
    f_avg_logprob = _mean(doc_logprobs) if doc_logprobs else 0.0
    f_std_logprob = _std(doc_logprobs) if doc_logprobs else 0.0
    f_low_perplexity_sentence_ratio = _safe_div(
        sum(1 for p in sent_perplexities if p < 20), len(sentences) or 1
    )
    f_log_rank_avg = _mean([abs(lp) for lp in doc_logprobs]) if doc_logprobs else 0.0

    f_coherence_mean, f_coherence_var = _sentence_similarity_features(sentences)

    f_avg_word_len = _mean(word_lens)
    f_avg_sent_len_words = _mean(sent_word_counts)
    f_avg_sent_len_chars = _mean(sent_char_lens)
    f_std_sent_len = _std(sent_word_counts)
    sorted_sents = sorted(sent_word_counts)
    top10_n = max(1, len(sorted_sents) // 10)
    top10_avg = _mean(sorted_sents[-top10_n:])
    rest_avg = _mean(sorted_sents[:-top10_n]) if len(sorted_sents) > top10_n else top10_avg
    f_top10_vs_rest_ratio = _safe_div(top10_avg, rest_avg or 1)
    f_sent_len_cv = _safe_div(_std(sent_word_counts), _mean(sent_word_counts) or 1)

    denom = n_words or 1
    f_comma_ratio = _safe_div(text.count(","), denom)
    total_punct = sum(1 for ch in text if ch in string.punctuation)
    f_total_punct_ratio = _safe_div(total_punct, denom)

    freqs = Counter(words_lower)
    probs = [c / n_words for c in freqs.values()] if n_words else []
    f_shannon_entropy = float(-sum(p * math.log2(p) for p in probs)) if probs else 0.0
    f_lexical_density = _safe_div(
        sum(1 for w in words_lower if w not in STOPWORDS), n_words or 1
    )
    rare_count = sum(1 for w in words_lower if 0 < zipf_frequency(w, "en") < 3.0)
    f_rare_word_ratio = _safe_div(rare_count, n_words or 1)
    f_top_word_freq_ratio = _safe_div(max(freqs.values()) if freqs else 0, n_words or 1)
    f_yules_k = _yules_k(words)
    f_mattr = _mattr(words)

    f_long_word_ratio = _safe_div(sum(1 for w in words if len(w) > 7), n_words or 1)
    f_short_word_ratio = _safe_div(sum(1 for w in words if len(w) <= 3), n_words or 1)

    verb_r, noun_r, adj_r, adv_r = _pos_ratios(words)
    f_verb_ratio = verb_r
    f_noun_ratio = noun_r
    f_adj_ratio = adj_r
    f_adv_ratio = adv_r
    f_passive_ratio = _passive_ratio(sentences)
    f_conjunction_ratio = _safe_div(
        sum(1 for w in words_lower if w in CONJUNCTIONS), n_words or 1
    )

    f_bigram_rep = _ngram_repetition_ratio(words_lower, 2)
    f_trigram_rep = _ngram_repetition_ratio(words_lower, 3)
    f_top_trigram_ratio = _top_ngram_ratio(words_lower, 3)

    f_stopword_ratio = _safe_div(
        sum(1 for w in words_lower if w in STOPWORDS), n_words or 1
    )
    f_readability = _flesch_reading_ease(words, sentences)
    f_parallelism_ratio = _structural_parallelism_ratio(sentences)

    return np.array([
        f_perplexity, f_perplexity_variance, f_avg_logprob, f_std_logprob,
        f_low_perplexity_sentence_ratio, f_log_rank_avg,
        f_coherence_mean, f_coherence_var,
        f_avg_word_len, f_avg_sent_len_words, f_avg_sent_len_chars,
        f_std_sent_len, f_top10_vs_rest_ratio, f_sent_len_cv,
        f_comma_ratio, f_total_punct_ratio,
        f_shannon_entropy, f_lexical_density, f_rare_word_ratio,
        f_top_word_freq_ratio, f_yules_k, f_mattr,
        f_long_word_ratio, f_short_word_ratio,
        f_verb_ratio, f_noun_ratio, f_adj_ratio, f_adv_ratio,
        f_passive_ratio, f_conjunction_ratio,
        f_bigram_rep, f_trigram_rep, f_top_trigram_ratio,
        f_stopword_ratio, f_readability, f_parallelism_ratio,
    ], dtype=float)





def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionScratch:
    def __init__(self, n_features=36):
        self.n_features = n_features
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.mean_ = None
        self.std_ = None
        self.class_weight_ = None

    def _fit_scaler(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0

    def _transform(self, X):
        return (X - self.mean_) / self.std_

    def _resolve_sample_weights(self, y, class_weight):

        m = len(y)
        if class_weight is None:
            self.class_weight_ = {0: 1.0, 1: 1.0}
            return np.ones(m, dtype=float)

        if class_weight == "balanced":
            n_pos = float(np.sum(y == 1))
            n_neg = float(np.sum(y == 0))
            n_pos = n_pos or 1.0
            n_neg = n_neg or 1.0
            w_pos = m / (2.0 * n_pos)
            w_neg = m / (2.0 * n_neg)
            self.class_weight_ = {0: w_neg, 1: w_pos}
        else:
            self.class_weight_ = dict(class_weight)

        sample_w = np.where(y == 1, self.class_weight_[1], self.class_weight_[0])
        return sample_w

    def fit(self, X, y, epochs=2000, lr=0.1, l2_lambda=0.0, class_weight=None,
            verbose=True, print_every=200):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        m = X.shape[0]

        self._fit_scaler(X)
        X = self._transform(X)

        sample_w = self._resolve_sample_weights(y, class_weight)
        sample_w = sample_w / sample_w.mean()

        self.w = np.zeros(self.n_features)
        self.b = 0.0

        for epoch in range(1, epochs + 1):
            z = np.dot(X, self.w) + self.b
            p = sigmoid(z)

            eps = 1e-9
            per_sample_loss = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            cost = np.mean(sample_w * per_sample_loss)
            cost += (l2_lambda / (2 * m)) * np.sum(self.w ** 2)

            dz = sample_w * (p - y)
            dw = np.dot(X.T, dz) / m + (l2_lambda / m) * self.w
            db = np.sum(dz) / m

            self.w -= lr * dw
            self.b -= lr * db

            if verbose and (epoch % print_every == 0 or epoch == 1):
                print(f"epoch {epoch:5d}  cost = {cost:.6f}")

        return self

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        x = self._transform(x)
        z = np.dot(x, self.w) + self.b
        p = sigmoid(z)
        return float(p[0]) if single else p

    def predict(self, x, threshold=0.5):
        p = self.predict_proba(x)
        if isinstance(p, float):
            return int(p >= threshold)
        return (p >= threshold).astype(int)

    def export_weights(self, path="model.json"):
        payload = {
            "w": self.w.tolist(),
            "b": float(self.b),
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }
        with open(path, "w") as f:
            json.dump(payload, f)
        return payload





RECORD_END_RE = re.compile(r',([01])(?=\r?\n|$)')


def robust_load_csv(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding, errors="replace") as f:
        raw = f.read()

    if not raw.strip():
        return pd.DataFrame(columns=["text", "label"])

    first_newline = raw.find("\n")
    first_line = raw[:first_newline if first_newline != -1 else len(raw)].strip().lower()
    body = raw[first_newline + 1:] if first_line.startswith("text") and "label" in first_line else raw

    records = []
    skipped_empty = 0
    pos = 0

    for m in RECORD_END_RE.finditer(body):
        chunk = body[pos:m.end()]
        pos = m.end()

        chunk = chunk.rstrip("\r\n")
        if len(chunk) < 2:
            skipped_empty += 1
            continue

        label = chunk[-1]
        text_part = chunk[:-2].strip()

        if text_part.startswith('"') and text_part.endswith('"') and len(text_part) >= 2:
            text_part = text_part[1:-1].replace('""', '"')

        text_part = text_part.strip()
        if not text_part:
            skipped_empty += 1
            continue

        records.append((text_part, int(label)))

    df = pd.DataFrame(records, columns=["text", "label"])
    print(f"Recovered {len(df)} valid records "
          f"(skipped {skipped_empty} empty/unusable fragments)")
    return df






DATA_PATH = "/kaggle/input/datasets/ysfa7med/mydata/fd.csv"

INPUT_FEATURES_CSV = "/kaggle/input/datasets/ysfa7med/features/extracted_features.csv"
WORKING_FEATURES_CSV = "/kaggle/working/extracted_features.csv"

df = robust_load_csv(DATA_PATH)
total_rows = len(df)
print(f"Loaded {total_rows} rows total.")

start_index = 0

if not os.path.exists(WORKING_FEATURES_CSV) and os.path.exists(INPUT_FEATURES_CSV):
    print("Copying existing features to working directory to resume writing...")
    shutil.copyfile(INPUT_FEATURES_CSV, WORKING_FEATURES_CSV)

if os.path.exists(WORKING_FEATURES_CSV):
    try:
        done_df = pd.read_csv(WORKING_FEATURES_CSV)
        start_index = len(done_df)
        print(f"Found existing features file in working directory. Resuming from row {start_index}...")
    except pd.errors.EmptyDataError:
        start_index = 0

if start_index < total_rows:
    print(f"Extracting features from row {start_index} to {total_rows}...")

    mode = 'a' if start_index > 0 else 'w'
    with open(WORKING_FEATURES_CSV, mode) as f:
        if start_index == 0:
            header = ["label"] + [f"f_{i}" for i in range(36)]
            f.write(",".join(header) + "\n")

        for i in range(start_index, total_rows):
            text = df["text"].iloc[i]
            label = df["label"].iloc[i]

            feats = features(text)

            row_data = [str(label)] + [str(x) for x in feats]
            f.write(",".join(row_data) + "\n")
            f.flush()

            if (i + 1) % 100 == 0:
                print(f"Successfully processed and saved up to row {i + 1}")

    print("Finished extracting all features!")

print("\nLoading all extracted features for model training...")
full_features_df = pd.read_csv(WORKING_FEATURES_CSV)

print("Dropping the first 6 GPT-2 features (Train-Serve Skew: no GPT-2 at the edge)")
print("and f_18 (rare_word_ratio: Cloudflare's wordFreq table can't cover the full")
print("long-tail rare-word range, so this feature is unreliable at inference time)...")
cols_to_drop = ["f_0", "f_1", "f_2", "f_3", "f_4", "f_5", "f_18"]
full_features_df = full_features_df.drop(columns=cols_to_drop, errors='ignore')

y = full_features_df["label"].to_numpy()
X = full_features_df.drop("label", axis=1).to_numpy()

print("New Feature matrix shape:", X.shape)
print(f"Class balance -> human(0): {(y == 0).sum()}   ai(1): {(y == 1).sum()}")

rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
split = int(0.95 * len(X))
train_idx, test_idx = idx[:split], idx[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

model = LogisticRegressionScratch(n_features=29)
print("Training Logistic Regression model (class_weight='balanced', l2_lambda=0.05)...")
model.fit(
    X_train, y_train,
    epochs=3000, lr=0.1,
    l2_lambda=0.05,

    class_weight="balanced",

)
print(f"Resolved class weights: {model.class_weight_}")

preds = model.predict(X_test)
acc = (preds == y_test).mean()
print(f"\nTest accuracy: {acc:.4f}")

feature_names = [
    "coherence_mean", "coherence_var", "avg_word_len", "avg_sent_len_words",
    "avg_sent_len_chars", "std_sent_len", "top10_vs_rest_ratio", "sent_len_cv",
    "comma_ratio", "total_punct_ratio", "shannon_entropy", "lexical_density",
    "top_word_freq_ratio", "yules_k", "mattr",
    "long_word_ratio", "short_word_ratio", "verb_ratio", "noun_ratio",
    "adj_ratio", "adv_ratio", "passive_ratio", "conjunction_ratio",
    "bigram_rep", "trigram_rep", "top_trigram_ratio", "stopword_ratio",
    "readability", "parallelism_ratio",
]
importance = sorted(zip(feature_names, model.w), key=lambda t: abs(t[1]), reverse=True)
print("\nTop features by |coefficient| (positive = pushes toward 'AI'):")
for name, coef in importance[:15]:
    print(f"  {name:22s}  {coef:+.4f}")

model.export_weights("model.json")
print("Exported: model.json")

proba_test = model.predict_proba(X_test)
avg_proba_label0 = proba_test[y_test == 0].mean()
avg_proba_label1 = proba_test[y_test == 1].mean()
print(f"avg P(class=1) label=0: {avg_proba_label0:.3f}")
print(f"avg P(class=1) label=1: {avg_proba_label1:.3f}")