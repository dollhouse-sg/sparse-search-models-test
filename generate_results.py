"""Run every sparse encoder over every language condition."""

from __future__ import annotations

import json
import math
import platform
import statistics
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Import the sibling module regardless of the working directory.
sys.path.insert(0, str(Path(__file__).parent))

from encoders import Encoder, EncodeStats, all_encoders

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results.json"
TOP_K = 10
N_TERMS = 10


def load_corpus() -> list[dict]:
    """Load corpus.json. Every chunk has an English and a Korean parallel."""
    corpus = json.loads((ROOT / "corpus.json").read_text(encoding="utf-8"))
    return corpus["chunks"]


CONDITIONS = [
    {
        "key": "full_en",
        "label": "English → English",
        "short": "EN → EN",
        "doc_lang": "en",
        "query_lang": "en",
        "description": "All 120 chunks. The monolingual baseline the cross-lingual conditions are compared against.",
    },
    {
        "key": "par_ko_ko",
        "label": "Korean → Korean",
        "short": "KO → KO",
        "doc_lang": "ko",
        "query_lang": "ko",
        "description": "Korean queries against Korean manual text. Monolingual, but tests whether each model handles Korean at all.",
    },
    {
        "key": "par_en_ko",
        "label": "English → Korean",
        "short": "EN → KO",
        "doc_lang": "ko",
        "query_lang": "en",
        "description": "English query, Korean documents. Matching can only happen through shared identifiers or genuine multilingual representation.",
    },
    {
        "key": "par_ko_en",
        "label": "Korean → English",
        "short": "KO → EN",
        "doc_lang": "en",
        "query_lang": "ko",
        "description": "Korean query, English documents. The reverse direction, which is the common case for a Korean operator searching English OEM docs.",
    },
]


def _rank_order(scores: np.ndarray) -> np.ndarray:
    """Doc indices by descending score, ties broken by position."""
    return np.lexsort((np.arange(len(scores)), -scores))


def rank_of_gold(scores: np.ndarray, gold_idx: int) -> int | None:
    """1-based rank of the gold doc, or None if nothing scored above zero.

    An all-zero score vector means the model found no shared dimension at all.
    Reporting a positional rank there would be luck, so we call it a miss.
    """
    if float(np.max(scores)) <= 0.0:
        return None
    pos = int(np.where(_rank_order(scores) == gold_idx)[0][0])
    return pos + 1


def top_hits(
    scores: np.ndarray, ids: list[str], gold_idx: int, k: int = 5
) -> list[dict]:
    """Top-k scored docs for the 'why did it match' panel, gold doc flagged."""
    return [
        {
            "id": ids[i],
            "score": round(float(scores[i]), 4),
            "gold": bool(i == gold_idx),
        }
        for i in _rank_order(scores)[:k]
    ]


def summarise(
    ranks: Sequence[int | None],
    tiers: Sequence[str],
    idgroups: Sequence[str],
    stats: EncodeStats,
    n_docs: int,
) -> dict[str, Any]:
    """Roll per-query ranks and encoder stats into one condition's metrics block."""
    ok = [r for r in ranks if r is not None]

    def hit(k: int) -> float:
        return round(sum(1 for r in ok if r <= k) / len(ranks), 4) if ranks else 0.0

    def hit_where(pred: Iterable[bool], k: int = 5) -> float | None:
        sel = [r for r, keep in zip(ranks, pred) if keep]
        if not sel:
            return None
        return round(sum(1 for r in sel if r is not None and r <= k) / len(sel), 4)

    mrr = (
        round(sum(1.0 / r for r in ok if r <= TOP_K) / len(ranks), 4) if ranks else 0.0
    )
    qms = stats.query_ms or [0.0]
    docs_per_s = round(n_docs / stats.index_s, 2) if stats.index_s > 0 else None
    mean_nnz = statistics.mean(stats.doc_nnz) if stats.doc_nnz else 0
    mean_in = statistics.mean(stats.doc_input_tokens) if stats.doc_input_tokens else 0
    p95 = min(len(qms) - 1, max(0, math.ceil(0.95 * len(qms)) - 1))  # nearest-rank
    return {
        "hit@1": hit(1),
        "hit@3": hit(3),
        "hit@5": hit(5),
        "hit@10": hit(10),
        "mrr@10": mrr,
        "misses": sum(1 for r in ranks if r is None),
        "mean_rank": round(statistics.mean(ok), 2) if ok else None,
        "by_tier": {
            t: hit_where([x == t for x in tiers]) for t in ("easy", "medium", "hard")
        },
        "by_identifier": {
            g: hit_where([x == g for x in idgroups]) for g in ("with_ids", "prose_only")
        },
        "index_s": round(stats.index_s, 3),
        "docs_per_s": docs_per_s,
        "query_ms_mean": round(statistics.mean(qms), 3),
        "query_ms_p95": round(sorted(qms)[p95], 3),
        "doc_nnz_mean": round(mean_nnz, 1),
        "expansion_ratio": round(mean_nnz / mean_in, 2) if mean_in else None,
        "query_nnz_mean": round(
            statistics.mean(stats.query_nnz) if stats.query_nnz else 0, 1
        ),
    }


def model_meta(enc: Encoder, error: str | None = None) -> dict[str, Any]:
    """Same keys whether or not the model loaded."""
    return {
        "key": enc.key,
        "label": enc.label,
        "family": enc.family,
        "multilingual": enc.multilingual,
        "params_m": enc.params_m,
        "notes": enc.notes,
        "load_s": None if error else round(enc.stats.load_s, 2),
        "inference_free_query": bool(getattr(enc, "has_query_weights", False)),
        "error": error,
    }


def main() -> int:
    """Load every model, run every condition, and write results.json."""
    chunks = load_corpus()
    print(f"corpus: {len(chunks)} chunks")

    # Load up front so an unreachable checkpoint is reported once, not per condition.
    meta: list[dict] = []
    live: list[Encoder] = []
    for enc in all_encoders():
        print(f"\nloading {enc.label} ...", flush=True)
        try:
            enc.load()
        except Exception as e:  # noqa: BLE001 - one bad model must not kill the run
            print(f"  FAILED: {type(e).__name__}: {e}")
            meta.append(model_meta(enc, error=f"{type(e).__name__}: {e}"))
            continue
        live.append(enc)
        meta.append(model_meta(enc))
        print(f"  ok  ({enc.stats.load_s:.1f}s, {enc.params_m or 0}M params)")

    conditions_out = []
    for cond in CONDITIONS:
        ids = [c["id"] for c in chunks]
        docs = [c["text_ko"] if cond["doc_lang"] == "ko" else c["text"] for c in chunks]
        queries = [
            c["query_ko"] if cond["query_lang"] == "ko" else c["query"] for c in chunks
        ]
        tiers = [c["tier"] for c in chunks]
        idgroups = ["with_ids" if c["identifiers"] else "prose_only" for c in chunks]

        print(f"\n=== {cond['label']}  ({len(docs)} docs) ===", flush=True)
        metrics: dict[str, Any] = {}
        per_query = [
            {
                "id": c["id"],
                "query": q,
                "tier": c["tier"],
                "vendor": c["vendor"],
                "bucket": c["bucket"],
                "identifiers": c["identifiers"],
                "per_model": {},
            }
            for c, q in zip(chunks, queries)
        ]

        for enc in live:
            enc.stats = EncodeStats()  # fresh timing per condition
            try:
                enc.index(docs)
                ranks: list[int | None] = []
                for gi, q in enumerate(queries):
                    s = enc.score(q)
                    r = rank_of_gold(s, gi)
                    ranks.append(r)
                    per_query[gi]["per_model"][enc.key] = {
                        "rank": r,
                        "top5": top_hits(s, ids, gi, 5),
                        "terms": [
                            [t, round(w, 4)] for t, w in enc.query_terms(q, N_TERMS)
                        ],
                    }
                metrics[enc.key] = summarise(
                    ranks, tiers, idgroups, enc.stats, len(docs)
                )
                m = metrics[enc.key]
                print(
                    f"  {enc.label:34s} hit@5={m['hit@5']:.3f} "
                    f"mrr={m['mrr@10']:.3f} miss={m['misses']:3d} "
                    f"idx={m['index_s']:6.1f}s q={m['query_ms_mean']:.1f}ms",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001 - one bad model must not kill the run
                print(f"  {enc.label}: FAILED {type(e).__name__}: {e}")
                metrics[enc.key] = {"error": f"{type(e).__name__}: {e}"}

        conditions_out.append(
            {
                **{k: v for k, v in cond.items() if k != "subset"},
                "n_docs": len(docs),
                "n_queries": len(queries),
                "metrics": metrics,
                "queries": per_query,
            }
        )

    try:
        import torch

        torch_v, threads = torch.__version__, torch.get_num_threads()
    except Exception:  # noqa: BLE001 - torch metadata is informational only
        torch_v, threads = None, None

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": {
            "cpu": platform.processor() or platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch_v,
            "torch_threads": threads,
        },
        "models": meta,
        "conditions": conditions_out,
        "corpus": [
            {
                k: c[k]
                for k in (
                    "id",
                    "vendor",
                    "bucket",
                    "tier",
                    "title",
                    "text",
                    "query",
                    "text_ko",
                    "query_ko",
                    "identifiers",
                )
            }
            for c in chunks
        ],
    }

    OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
