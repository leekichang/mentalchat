# MentalChat16K Clustering

This project provides a single Python script to:

1. Download `ShenLab/MentalChat16K` from Hugging Face.
2. Run **input-only** text clustering with:
   - TF-IDF + KMeans
   - Pretrained embedding vectors (GloVe via `gensim`) + KMeans
3. Generate 2D visualizations:
   - TF-IDF: t-SNE and UMAP
   - Embeddings: UMAP
4. Cache embedding vectors in a space-efficient format (`float16` compressed `.npz`) and reuse them on later runs.

## File

- `cluster_mentalchat16k.py`

## Requirements

- Python 3.10+ (3.11 recommended)

Install dependencies:

```bash
python -m pip install datasets pandas numpy scikit-learn matplotlib umap-learn gensim tqdm pyarrow
```

## Quick Start

Run on the full dataset:

```bash
python cluster_mentalchat16k.py \
  --k 8 \
  --max-samples 0 \
  --viz-max-samples 3000 \
  --output-dir outputs_full \
  --cache-dir cache_full
```

Run a smaller test:

```bash
python cluster_mentalchat16k.py \
  --k 5 \
  --max-samples 500 \
  --viz-max-samples 500 \
  --output-dir outputs_test \
  --cache-dir cache_test
```

## Default Behavior

- If `input` column exists, clustering uses `input` only.
- If `input` does not exist, text columns are auto-detected.
- Embeddings are generated from a pretrained model:
  - default: `glove-wiki-gigaword-100`

## Output Files

Main outputs (under `--output-dir`):

- `mentalchat16k_clustered_tfidf.csv`
- `cluster_summary_tfidf.csv`
- `cluster_scatter_tfidf_tsne.png`
- `cluster_scatter_tfidf_umap.png`
- `cluster_points_tfidf_tsne.csv`
- `cluster_points_tfidf_umap.csv`
- `mentalchat16k_clustered_embed.csv`
- `cluster_summary_embed.csv`
- `cluster_scatter_embed_umap.png`
- `cluster_points_embed_umap.csv`

Cache outputs (under `--cache-dir`):

- `embeddings_<hash>.npz` (compressed `float16` vectors)
- `embeddings_<hash>.json` (cache metadata)
- `gensim-data/` (downloaded pretrained embedding resources)

## Progress Monitoring

The script uses `tqdm` progress bars for:

- row materialization (`[data] rows`)
- embedding batch encoding (`[embed] encoding`)

## Key Options

```bash
python cluster_mentalchat16k.py --help
```

Common options:

- `--k`: number of clusters
- `--max-samples`: cap number of rows (0 = all)
- `--viz-max-samples`: cap rows used for 2D plotting (0 = all)
- `--embedding-model`: pretrained embedding name from `gensim.downloader`
- `--embed-batch-size`: batch size for document embedding
- `--skip-embedding`: run TF-IDF pipeline only
- `--text-columns`: override text columns manually

## Notes

- First embedding run may download pretrained vectors; subsequent runs reuse cached vectors.
- UMAP warning about `n_jobs` with fixed `random_state` is expected.
