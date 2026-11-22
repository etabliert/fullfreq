from datasets import load_dataset
import itertools
import json

fw = load_dataset("HuggingFaceFW/fineweb-2", name="deu_Latn", split="train", streaming=True)

with open("fineweb_data_de.jsonl", "w", encoding="utf-8") as f:
    for doc in itertools.islice(fw, 50000):
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
