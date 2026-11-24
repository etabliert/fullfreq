from datasets import load_dataset
import itertools
import json

# config
documents = 50000
lang = "de"     # only changes writed document lang, fineweb uses another language code, idk how to convert.

fw = load_dataset("HuggingFaceFW/fineweb-2", name="deu_Latn", split="train", streaming=True)

with open(f"fineweb_data_{lang}.jsonl", "w", encoding="utf-8") as f:
    for doc in itertools.islice(fw, documents):
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
