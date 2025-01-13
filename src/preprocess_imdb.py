import polars as pl

imdb = pl.scan_parquet(
    [
        "hf://datasets/stanfordnlp/imdb/plain_text/test-00000-of-00001.parquet",
        "hf://datasets/stanfordnlp/imdb/plain_text/train-00000-of-00001.parquet",
    ],
)

# Remove br elements and sample 5000 observations
imdb_cleaned = (
    imdb.with_columns(pl.col("text").str.replace_all("<br />", "\n", literal=True))
    .collect()
    .sample(10_000)
)

imdb_cleaned.write_parquet("imdb.parquet", compression_level=1)
