import json
from tqdm import tqdm
from datatrove.pipeline.readers import ParquetReader

# config
documents = 50000
lang = "de"
langFW = "deu_Latn"     # fineweb-2 uses a diffrent layout, go to https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 to find how it is for other languages

data_reader = ParquetReader(f"hf://datasets/HuggingFaceFW/fineweb-2/data/{langFW}/train", limit=documents) 

with open(f"fineweb_data_{lang}.jsonl", "w", encoding="utf-8") as f:
    for doc in tqdm(data_reader(), total=documents):
        f.write(json.dumps({"text": doc.text}, ensure_ascii=False) + "\n")