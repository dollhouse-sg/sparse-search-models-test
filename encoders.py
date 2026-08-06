"""Sparse retrieval encoders with a uniform interface."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    import torch

_log = logging.getLogger(__name__)

# Identifiers must survive tokenisation intact: S7-1500, F07801, 250, N-m.
_WORD = re.compile(r"[a-z0-9][a-z0-9._/+-]*")
_HANGUL = re.compile(r"[가-힯]+")


def lexical_tokens(text: str) -> list[str]:
    """Lowercase word/identifier tokens plus Korean character bigrams."""
    text = text.lower()
    toks = _WORD.findall(text)
    for run in _HANGUL.findall(text):
        if len(run) == 1:
            toks.append(run)
        else:
            toks.extend(run[i : i + 2] for i in range(len(run) - 1))
    return toks


@dataclass
class EncodeStats:
    """Timing and sparsity numbers collected during load/index/score."""

    load_s: float = 0.0
    index_s: float = 0.0
    query_ms: list[float] = field(default_factory=list)
    doc_nnz: list[int] = field(default_factory=list)
    query_nnz: list[int] = field(default_factory=list)
    doc_input_tokens: list[int] = field(default_factory=list)


class Encoder:
    """Base interface."""

    key: str = ""
    label: str = ""
    family: str = ""  # "lexical" | "learned"
    multilingual: bool = False
    params_m: float | None = None
    notes: str = ""

    def __init__(self) -> None:
        self.stats = EncodeStats()

    def load(self) -> None:
        pass

    def index(self, docs: list[str]) -> None:
        raise NotImplementedError

    def score(self, query: str) -> np.ndarray:
        """One score per indexed doc, in the order given to index()."""
        raise NotImplementedError

    def query_terms(self, query: str, top_k: int = 12) -> list[tuple[str, float]]:
        """Highest weighted query dimensions, for the 'why did it match' panel."""
        return []


# --------------------------------------------------------------------------- #
# Lexical
# --------------------------------------------------------------------------- #


class BM25Encoder(Encoder):
    key = "bm25"
    label = "BM25"
    family = "lexical"
    multilingual = False
    notes = "Okapi BM25 over word tokens plus Korean character bigrams."

    def index(self, docs: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        t0 = time.perf_counter()
        self._toks = [lexical_tokens(d) for d in docs]
        self._bm25 = BM25Okapi(self._toks)
        self.stats.index_s = time.perf_counter() - t0
        for t in self._toks:
            self.stats.doc_nnz.append(len(set(t)))
            self.stats.doc_input_tokens.append(len(set(t)))

    def score(self, query: str) -> np.ndarray:
        t0 = time.perf_counter()
        q = lexical_tokens(query)
        s = self._bm25.get_scores(q)
        self.stats.query_ms.append((time.perf_counter() - t0) * 1000)
        self.stats.query_nnz.append(len(set(q)))
        return np.asarray(s, dtype=np.float64)

    def query_terms(self, query: str, top_k: int = 12) -> list[tuple[str, float]]:
        # No query-side weights, so fall back to term frequency.
        seen: dict[str, float] = {}
        for t in lexical_tokens(query):
            seen[t] = seen.get(t, 0.0) + 1.0
        return sorted(seen.items(), key=lambda kv: -kv[1])[:top_k]


class TfidfEncoder(Encoder):
    key = "tfidf"
    label = "TF-IDF"
    family = "lexical"
    multilingual = False
    notes = "Cosine similarity over TF-IDF vectors, same tokenisation as BM25."

    def index(self, docs: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        t0 = time.perf_counter()
        self._vec = TfidfVectorizer(analyzer=lexical_tokens)
        self._mat = self._vec.fit_transform(docs)
        # L2 normalised by default, so dot product is cosine.
        self.stats.index_s = time.perf_counter() - t0
        nnz = self._mat.getnnz(axis=1)
        self.stats.doc_nnz.extend(int(x) for x in nnz)
        self.stats.doc_input_tokens.extend(int(x) for x in nnz)

    def score(self, query: str) -> np.ndarray:
        t0 = time.perf_counter()
        qv = self._vec.transform([query])
        s = (self._mat @ qv.T).toarray().ravel()
        self.stats.query_ms.append((time.perf_counter() - t0) * 1000)
        self.stats.query_nnz.append(int(qv.getnnz()))
        return s

    def query_terms(self, query: str, top_k: int = 12) -> list[tuple[str, float]]:
        qv = self._vec.transform([query]).tocoo()
        names = self._vec.get_feature_names_out()
        pairs = [(names[j], float(v)) for j, v in zip(qv.col, qv.data)]
        return sorted(pairs, key=lambda kv: -kv[1])[:top_k]


# --------------------------------------------------------------------------- #
# Learned sparse
# --------------------------------------------------------------------------- #


class MlmSparseEncoder(Encoder):
    """SPLADE-style: sparse weights read off a masked-LM head over the vocab.

    doc vector = pool_max( transform(logits) * attention_mask )

    SPLADE uses log(1+relu(x)); the OpenSearch models use the same formulation.
    Because the head runs over the whole vocabulary these models *expand*: they
    assign weight to terms that never appeared in the input.

    With inference_free_query the query side skips the model and looks token
    weights up in a table shipped with the checkpoint.
    """

    family = "learned"
    transform = "log1p_relu"
    max_length = 512
    batch_size = 8
    inference_free_query = False

    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.model_id = model_id
        self._mat: sparse.csr_matrix | None = None
        self._qw: dict[int, float] | None = None

    @property
    def has_query_weights(self) -> bool:
        return self._qw is not None

    def load(self) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_id)
        self.model.eval()
        self.params_m = round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1)
        if self.inference_free_query:
            self._qw = self._load_query_weights()
            if self._qw is None:
                # Not fatal, but the run loses the latency advantage.
                _log.warning(
                    "%s: no query weight table, falling back to model inference "
                    "on the query side",
                    self.model_id,
                )
        self.stats.load_s = time.perf_counter() - t0

    def _load_query_weights(self) -> dict[int, float] | None:
        """Token weight lookup used by the inference-free query side."""
        from huggingface_hub import hf_hub_download

        for fname in ("query_token_weights.txt", "idf.json"):
            try:
                path = hf_hub_download(self.model_id, fname)
            except Exception as e:  # noqa: BLE001 - hub errors vary; try the next filename
                _log.debug("could not fetch %s for %s: %s", fname, self.model_id, e)
                continue
            try:
                weights = self._parse_query_weights(path, fname)
            except Exception as e:  # noqa: BLE001 - malformed file; try the next one
                _log.debug("could not parse %s for %s: %s", fname, self.model_id, e)
                continue
            if weights:
                return weights
        return None

    def _parse_query_weights(self, path: str, fname: str) -> dict[int, float]:
        if fname.endswith(".json"):
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            return {self.tok.convert_tokens_to_ids(k): float(v) for k, v in raw.items()}
        weights: dict[int, float] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    weights[self.tok.convert_tokens_to_ids(parts[0])] = float(parts[1])
        return weights

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        import torch

        if self.transform == "log1p_relu":
            return torch.log1p(torch.relu(logits))
        return torch.relu(logits)

    def _encode(self, texts: list[str]) -> sparse.csr_matrix:
        import torch

        rows = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                enc = self.tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                out = self.model(**enc).logits  # (B, L, V)
                w = self._apply(out)
                w = w * enc["attention_mask"].unsqueeze(-1)
                vec = w.max(dim=1).values  # (B, V)
                rows.append(sparse.csr_matrix(vec.numpy()))
        return sparse.vstack(rows).tocsr()

    def index(self, docs: list[str]) -> None:
        t0 = time.perf_counter()
        self._mat = self._encode(docs)
        self.stats.index_s = time.perf_counter() - t0
        self.stats.doc_nnz.extend(int(x) for x in self._mat.getnnz(axis=1))
        for d in docs:
            ids = self.tok(d, truncation=True, max_length=self.max_length)["input_ids"]
            self.stats.doc_input_tokens.append(len(set(ids)))

    def _vocab_size(self) -> int:
        if self._mat is not None:
            return self._mat.shape[1]
        return int(self.model.config.vocab_size)

    def _query_vec(self, query: str) -> sparse.csr_matrix:
        if self._qw is None:
            return self._encode([query])
        ids = self.tok(query, truncation=True, max_length=self.max_length)["input_ids"]
        cols, vals = [], []
        for tid in set(ids):
            w = self._qw.get(tid)
            if w:
                cols.append(tid)
                vals.append(w)
        return sparse.csr_matrix(
            (vals, ([0] * len(cols), cols)),
            shape=(1, self._vocab_size()),
            dtype=np.float64,
        )

    def score(self, query: str) -> np.ndarray:
        t0 = time.perf_counter()
        qv = self._query_vec(query)
        s = (self._mat @ qv.T).toarray().ravel()
        self.stats.query_ms.append((time.perf_counter() - t0) * 1000)
        self.stats.query_nnz.append(int(qv.getnnz()))
        return s

    def query_terms(self, query: str, top_k: int = 12) -> list[tuple[str, float]]:
        qv = self._query_vec(query).tocoo()
        pairs = [
            (self.tok.convert_ids_to_tokens(int(j)), float(v))
            for j, v in zip(qv.col, qv.data)
        ]
        return sorted(pairs, key=lambda kv: -kv[1])[:top_k]


class SpladeEncoder(MlmSparseEncoder):
    key = "splade"
    label = "SPLADE++ ensembledistil"
    multilingual = False
    notes = (
        "BERT-base MLM head, log(1+ReLU) max-pooled. Expands into related English "
        "terms, which is why it beats BM25 on paraphrased queries."
    )

    def __init__(self) -> None:
        super().__init__("naver/splade-cocondenser-ensembledistil")


class OpenSearchEnEncoder(MlmSparseEncoder):
    key = "opensearch_en"
    label = "OpenSearch sparse v3 (EN)"
    multilingual = False
    inference_free_query = True
    notes = (
        "DistilBERT doc encoder, Apache-2.0. Query side is inference-free: a "
        "tokeniser plus a token weight lookup, so query latency is negligible."
    )

    def __init__(self) -> None:
        super().__init__(
            "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill"
        )


class OpenSearchMultiEncoder(MlmSparseEncoder):
    key = "opensearch_multi"
    label = "OpenSearch sparse multilingual"
    multilingual = True
    inference_free_query = True
    notes = (
        "160M multilingual learned-sparse encoder, Apache-2.0. The only expanding "
        "sparse model here that was trained on Korean."
    )

    def __init__(self) -> None:
        super().__init__(
            "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
        )


class BgeM3SparseEncoder(Encoder):
    """BGE-M3's sparse head only.

    Unlike SPLADE this does NOT expand: sparse_linear produces one weight per
    *input* token, scatter-maxed onto that token's vocab id. Cross-lingual
    matching therefore depends entirely on token ids shared between the two
    languages, which in OEM manuals means the identifiers.
    """

    key = "bgem3"
    label = "BGE-M3 (sparse head)"
    family = "learned"
    multilingual = True
    notes = (
        "XLM-RoBERTa-large backbone with the sparse_linear head. Weights input "
        "tokens only, no vocabulary expansion."
    )
    max_length = 512
    batch_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.model_id = "BAAI/bge-m3"

    def load(self) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer

        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.eval()
        # 3.5 kB Linear(hidden, 1) shipped alongside the backbone.
        path = hf_hub_download(self.model_id, "sparse_linear.pt")
        state = torch.load(path, map_location="cpu")
        hidden = self.model.config.hidden_size
        self.sparse_linear = torch.nn.Linear(hidden, 1)
        self.sparse_linear.load_state_dict(state)
        self.sparse_linear.eval()
        self.vocab_size = int(self.model.config.vocab_size)
        self.params_m = round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1)
        self.stats.load_s = time.perf_counter() - t0

    def _encode(self, texts: list[str]) -> sparse.csr_matrix:
        import torch

        rows = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                enc = self.tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                hidden = self.model(**enc).last_hidden_state
                w = torch.relu(self.sparse_linear(hidden)).squeeze(-1)  # (B, L)
                w = w * enc["attention_mask"]
                ids = enc["input_ids"]
                out = torch.zeros(ids.shape[0], self.vocab_size, dtype=w.dtype)
                # scatter-max: duplicate tokens keep their strongest weight
                out.scatter_reduce_(1, ids, w, reduce="amax", include_self=True)
                # special tokens carry no lexical meaning
                for sid in self.tok.all_special_ids:
                    out[:, sid] = 0.0
                rows.append(sparse.csr_matrix(out.numpy()))
        return sparse.vstack(rows).tocsr()

    def index(self, docs: list[str]) -> None:
        t0 = time.perf_counter()
        self._mat = self._encode(docs)
        self.stats.index_s = time.perf_counter() - t0
        self.stats.doc_nnz.extend(int(x) for x in self._mat.getnnz(axis=1))
        for d in docs:
            ids = self.tok(d, truncation=True, max_length=self.max_length)["input_ids"]
            self.stats.doc_input_tokens.append(len(set(ids)))

    def score(self, query: str) -> np.ndarray:
        t0 = time.perf_counter()
        qv = self._encode([query])
        s = (self._mat @ qv.T).toarray().ravel()
        self.stats.query_ms.append((time.perf_counter() - t0) * 1000)
        self.stats.query_nnz.append(int(qv.getnnz()))
        return s

    def query_terms(self, query: str, top_k: int = 12) -> list[tuple[str, float]]:
        qv = self._encode([query]).tocoo()
        pairs = [
            (self.tok.convert_ids_to_tokens(int(j)), float(v))
            for j, v in zip(qv.col, qv.data)
        ]
        return sorted(pairs, key=lambda kv: -kv[1])[:top_k]


def all_encoders() -> list[Encoder]:
    return [
        BM25Encoder(),
        TfidfEncoder(),
        SpladeEncoder(),
        OpenSearchEnEncoder(),
        OpenSearchMultiEncoder(),
        BgeM3SparseEncoder(),
    ]
