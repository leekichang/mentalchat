#!/usr/bin/env python3
"""
Quick starter script:
1) Download ShenLab/MentalChat16K from Hugging Face.
2) Run TF-IDF + KMeans clustering (input-only by default).
3) Run pretrained language model embedding + KMeans clustering.
4) Save summaries and 2D visualizations (t-SNE + UMAP where applicable).
5) Cache embedding vectors (float16 compressed) and reuse if available.

Example:
  python cluster_mentalchat16k.py --k 8 --max-samples 8000 --output-dir outputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from datasets import load_dataset
import gensim.downloader as gensim_api
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MentalChat16K and run TF-IDF / embedding clustering."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ShenLab/MentalChat16K",
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train).",
    )
    parser.add_argument(
        "--text-columns",
        nargs="+",
        default=None,
        help=(
            "Text columns to concatenate. "
            "If omitted, uses input-only when available."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of KMeans clusters for both TF-IDF and embeddings.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If > 0, limit rows for faster experiments.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=10000,
        help="Max TF-IDF vocabulary size.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=3,
        help="Min document frequency for TF-IDF terms.",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.95,
        help="Max document frequency for TF-IDF terms.",
    )
    parser.add_argument(
        "--top-terms",
        type=int,
        default=12,
        help="Top TF-IDF terms to show per cluster.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--viz-max-samples",
        type=int,
        default=3000,
        help="Max points to use for visualization (0 means all).",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity.",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="glove-wiki-gigaword-100",
        help="Pretrained embedding name from gensim.downloader.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=256,
        help="Batch size for document embedding conversion.",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding-based clustering.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Directory for embedding cache files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory.",
    )
    return parser.parse_args()


def to_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def detect_text_columns(sample_row: dict, provided: list[str] | None) -> list[str]:
    if provided:
        return provided

    if "input" in sample_row:
        return ["input"]

    preferred = [
        "text",
        "utterance",
        "dialogue",
        "prompt",
        "instruction",
        "output",
        "response",
        "content",
    ]
    keys = list(sample_row.keys())

    selected = [k for k in preferred if k in keys]
    if selected:
        return selected

    selected = []
    for key in keys:
        v = sample_row[key]
        if isinstance(v, str):
            selected.append(key)
        elif isinstance(v, list) and v and all(isinstance(x, str) for x in v[:3]):
            selected.append(key)
        elif isinstance(v, dict):
            selected.append(key)
    return selected


def build_texts(rows: Iterable[dict], columns: list[str]) -> list[str]:
    texts: list[str] = []
    for row in rows:
        pieces = [to_str(row.get(c, "")) for c in columns]
        merged = " ".join(p for p in pieces if p).strip()
        texts.append(merged)
    return texts


def top_terms_per_cluster(
    model: KMeans, vectorizer: TfidfVectorizer, top_n: int
) -> dict[int, list[str]]:
    terms = np.array(vectorizer.get_feature_names_out())
    out: dict[int, list[str]] = {}
    for i, center in enumerate(model.cluster_centers_):
        top_idx = np.argsort(center)[::-1][:top_n]
        out[i] = terms[top_idx].tolist()
    return out


def choose_indices(n: int, max_samples: int, seed: int) -> np.ndarray:
    if max_samples and max_samples > 0 and n > max_samples:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_samples, replace=False))
        return idx
    return np.arange(n)


def run_tsne(vectors: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    max_perplexity = max(5.0, min(50.0, (vectors.shape[0] - 1) / 3))
    use_perplexity = min(perplexity, max_perplexity)
    return TSNE(
        n_components=2,
        random_state=seed,
        perplexity=use_perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
    ).fit_transform(vectors)


def run_umap(
    vectors: np.ndarray, seed: int, n_neighbors: int, min_dist: float
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
    )
    return reducer.fit_transform(vectors)


def save_scatter_plot(
    emb2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    png_path: Path,
) -> None:
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        emb2d[:, 0],
        emb2d[:, 1],
        c=labels,
        cmap="tab10",
        s=10,
        alpha=0.8,
    )
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    legend = plt.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    plt.gca().add_artist(legend)
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()


def save_points_csv(
    idx: np.ndarray,
    labels: np.ndarray,
    emb2d: np.ndarray,
    csv_path: Path,
) -> None:
    pd.DataFrame(
        {
            "row_index": idx,
            "cluster": labels,
            "x": emb2d[:, 0],
            "y": emb2d[:, 1],
        }
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")


def get_embedding_cache_paths(
    cache_dir: Path,
    dataset: str,
    split: str,
    text_columns: list[str],
    model_name: str,
    n_rows: int,
) -> tuple[Path, Path]:
    cfg = {
        "dataset": dataset,
        "split": split,
        "text_columns": text_columns,
        "model_name": model_name,
        "n_rows": n_rows,
    }
    key = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    cache_npz = cache_dir / f"embeddings_{key}.npz"
    cache_meta = cache_dir / f"embeddings_{key}.json"
    return cache_npz, cache_meta


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def encode_embeddings_with_cache(
    texts: list[str],
    model_name: str,
    batch_size: int,
    cache_dir: Path,
    dataset: str,
    split: str,
    text_columns: list[str],
) -> tuple[np.ndarray, Path, bool]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_npz, cache_meta = get_embedding_cache_paths(
        cache_dir=cache_dir,
        dataset=dataset,
        split=split,
        text_columns=text_columns,
        model_name=model_name,
        n_rows=len(texts),
    )

    if cache_npz.exists():
        arr = np.load(cache_npz)["embeddings"].astype(np.float32)
        return arr, cache_npz, True

    gensim_dir = cache_dir / "gensim-data"
    gensim_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GENSIM_DATA_DIR"] = str(gensim_dir.resolve())

    model = gensim_api.load(model_name)
    vec_dim = int(model.vector_size)
    chunks = []
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="[embed] encoding",
        unit="batch",
    ):
        batch = texts[start : start + batch_size]
        batch_vec = np.zeros((len(batch), vec_dim), dtype=np.float32)
        for i, text in enumerate(batch):
            tokens = tokenize_text(text)
            vecs = [model[w] for w in tokens if w in model]
            if vecs:
                v = np.mean(vecs, axis=0)
                norm = float(np.linalg.norm(v))
                if norm > 0:
                    v = v / norm
                batch_vec[i] = v.astype(np.float32)
        chunks.append(batch_vec)

    embeddings = np.vstack(chunks)
    np.savez_compressed(cache_npz, embeddings=embeddings.astype(np.float16))
    cache_meta.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "split": split,
                "text_columns": text_columns,
                "model_name": model_name,
                "n_rows": len(texts),
                "cache_dtype": "float16",
                "shape": list(embeddings.shape),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return embeddings, cache_npz, False


def representative_text_summary(
    labels: np.ndarray,
    vectors: np.ndarray,
    texts: list[str],
) -> pd.DataFrame:
    rows = []
    cluster_ids = sorted(set(labels.tolist()))
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        cluster_vecs = vectors[idx]
        centroid = cluster_vecs.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(cluster_vecs - centroid, axis=1)
        best_pos = int(np.argmin(dists))
        best_idx = int(idx[best_pos])
        preview = texts[best_idx].replace("\n", " ").strip()[:200]
        rows.append(
            {
                "cluster": int(cid),
                "size": int(len(idx)),
                "repr_row_index": best_idx,
                "repr_text_preview": preview,
            }
        )
    return pd.DataFrame(rows).sort_values("cluster")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    print(f"[1/8] Loading dataset: {args.dataset} (split={args.split})")
    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples and args.max_samples > 0:
        n = min(args.max_samples, len(ds))
        ds = ds.select(range(n))
        print(f"  - Using subset: {n} rows")
    else:
        print(f"  - Rows: {len(ds)}")

    sample_row = ds[0]
    text_columns = detect_text_columns(sample_row, args.text_columns)
    if not text_columns:
        raise ValueError("No text-like columns detected. Pass --text-columns explicitly.")
    print(f"[2/8] Text columns: {text_columns}")

    print("[3/8] Materializing rows and merged texts")
    rows = [ds[i] for i in tqdm(range(len(ds)), desc="[data] rows", unit="row")]
    texts = build_texts(rows, text_columns)
    non_empty = sum(1 for t in texts if t)
    print(f"  - Non-empty merged texts: {non_empty}/{len(texts)}")
    if non_empty < 2:
        raise ValueError("Not enough non-empty texts for clustering.")

    print("[4/8] TF-IDF + KMeans clustering")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
    )
    X_tfidf = vectorizer.fit_transform(texts)
    k = min(args.k, len(texts))
    if k < 2:
        raise ValueError("k must be >= 2 after adjustment.")

    model_tfidf = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
    labels_tfidf = model_tfidf.fit_predict(X_tfidf)
    print(f"  - TF-IDF matrix shape: {X_tfidf.shape}")

    print("[5/8] Saving TF-IDF results")
    df = pd.DataFrame(rows)
    df["merged_text"] = texts
    df["cluster_tfidf"] = labels_tfidf
    tfidf_csv = output_dir / "mentalchat16k_clustered_tfidf.csv"
    df.to_csv(tfidf_csv, index=False, encoding="utf-8-sig")

    tfidf_top_terms = top_terms_per_cluster(model_tfidf, vectorizer, args.top_terms)
    tfidf_sizes = pd.Series(labels_tfidf).value_counts().sort_index()
    tfidf_summary = pd.DataFrame(
        {
            "cluster": list(range(k)),
            "size": [int(tfidf_sizes.get(i, 0)) for i in range(k)],
            "top_terms": [", ".join(tfidf_top_terms[i]) for i in range(k)],
        }
    )
    tfidf_summary_csv = output_dir / "cluster_summary_tfidf.csv"
    tfidf_summary.to_csv(tfidf_summary_csv, index=False, encoding="utf-8-sig")

    print("[6/8] TF-IDF visualization (t-SNE + UMAP)")
    idx_viz = choose_indices(len(texts), args.viz_max_samples, args.seed)
    X_tfidf_viz = X_tfidf[idx_viz]
    labels_tfidf_viz = labels_tfidf[idx_viz]
    svd_dim = min(50, max(2, X_tfidf_viz.shape[1] - 1))
    X_tfidf_viz_dense = TruncatedSVD(n_components=svd_dim, random_state=args.seed).fit_transform(
        X_tfidf_viz
    )

    tfidf_tsne_2d = run_tsne(X_tfidf_viz_dense, args.seed, args.tsne_perplexity)
    tfidf_tsne_png = output_dir / "cluster_scatter_tfidf_tsne.png"
    tfidf_tsne_pts = output_dir / "cluster_points_tfidf_tsne.csv"
    save_scatter_plot(
        emb2d=tfidf_tsne_2d,
        labels=labels_tfidf_viz,
        title="MentalChat16K TF-IDF Clusters (t-SNE)",
        png_path=tfidf_tsne_png,
    )
    save_points_csv(idx_viz, labels_tfidf_viz, tfidf_tsne_2d, tfidf_tsne_pts)

    tfidf_umap_2d = run_umap(
        X_tfidf_viz_dense,
        seed=args.seed,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )
    tfidf_umap_png = output_dir / "cluster_scatter_tfidf_umap.png"
    tfidf_umap_pts = output_dir / "cluster_points_tfidf_umap.csv"
    save_scatter_plot(
        emb2d=tfidf_umap_2d,
        labels=labels_tfidf_viz,
        title="MentalChat16K TF-IDF Clusters (UMAP)",
        png_path=tfidf_umap_png,
    )
    save_points_csv(idx_viz, labels_tfidf_viz, tfidf_umap_2d, tfidf_umap_pts)

    embed_csv = None
    embed_summary_csv = None
    embed_umap_png = None
    embed_cache_file = None
    embed_cache_hit = None

    if not args.skip_embedding:
        print("[7/8] Embedding + KMeans clustering")
        X_embed, embed_cache_file, embed_cache_hit = encode_embeddings_with_cache(
            texts=texts,
            model_name=args.embedding_model,
            batch_size=args.embed_batch_size,
            cache_dir=cache_dir,
            dataset=args.dataset,
            split=args.split,
            text_columns=text_columns,
        )
        print(
            f"  - Embedding shape: {X_embed.shape} | cache: {embed_cache_file} "
            f"| hit: {embed_cache_hit}"
        )
        model_embed = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
        labels_embed = model_embed.fit_predict(X_embed)

        df["cluster_embed"] = labels_embed
        embed_csv = output_dir / "mentalchat16k_clustered_embed.csv"
        df.to_csv(embed_csv, index=False, encoding="utf-8-sig")

        embed_summary = representative_text_summary(
            labels=labels_embed,
            vectors=X_embed,
            texts=texts,
        )
        embed_summary_csv = output_dir / "cluster_summary_embed.csv"
        embed_summary.to_csv(embed_summary_csv, index=False, encoding="utf-8-sig")

        idx_embed_viz = choose_indices(len(texts), args.viz_max_samples, args.seed)
        embed_umap_2d = run_umap(
            X_embed[idx_embed_viz],
            seed=args.seed,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )
        embed_umap_png = output_dir / "cluster_scatter_embed_umap.png"
        embed_umap_pts = output_dir / "cluster_points_embed_umap.csv"
        save_scatter_plot(
            emb2d=embed_umap_2d,
            labels=labels_embed[idx_embed_viz],
            title="MentalChat16K Embedding Clusters (UMAP)",
            png_path=embed_umap_png,
        )
        save_points_csv(
            idx=idx_embed_viz,
            labels=labels_embed[idx_embed_viz],
            emb2d=embed_umap_2d,
            csv_path=embed_umap_pts,
        )
    else:
        print("[7/8] Embedding stage skipped (--skip-embedding)")

    print("[8/8] Done")
    print(f"  - Saved: {tfidf_csv}")
    print(f"  - Saved: {tfidf_summary_csv}")
    print(f"  - Saved: {tfidf_tsne_png}")
    print(f"  - Saved: {tfidf_umap_png}")
    if embed_csv:
        print(f"  - Saved: {embed_csv}")
    if embed_summary_csv:
        print(f"  - Saved: {embed_summary_csv}")
    if embed_umap_png:
        print(f"  - Saved: {embed_umap_png}")


if __name__ == "__main__":
    main()
