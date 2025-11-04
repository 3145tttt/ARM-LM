```
bash download.sh
```

```
conda env create -f requirements.yml -n nlp
```

```
bash all_runs.sh
```

| Vocab size | Embedding dim | Layers | Validation Perplexity 
| ---- | ---- | ---- | ----
| 88 | 768 | 12 | **1.205**
| 88 | 384 | 6 | 1.305
| 200 | 768 | 12 | 1.9
| 200 | 384 | 6 | 2.055


[HF checkpoint](https://huggingface.co/3145tttt/arm_gpt_GPT2_88_12_768)
