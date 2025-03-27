# Using Encoder-Only Transformers for Sentiment Analysis

## Reproduce

You can use the Dockerfile. Just build it with

```
docker build . -t eotfsa
```

and run it. Running without any arguments will only do the preprocessing step. To get
dropped in a shell with all dependencies installed, use

```
docker run --rm -it eotfsa:latest bash -l
```
