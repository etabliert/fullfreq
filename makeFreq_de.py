import stanza
import json
from tqdm import tqdm
import csv
from collections import Counter
import re

dataset = "fineweb_data_de.jsonl"
nlp = stanza.Pipeline("de", processors="tokenize,pos", use_gpu=True, pos_batch_size=2000, tokenize_batch_size=128, verbose=False)

length = sum(1 for l in open(dataset, encoding="utf-8"))

freq = Counter()
batch = []

for i, line in enumerate(tqdm(open(dataset, encoding="utf-8"), total=length)):
    e = json.loads(line)
    text = re.sub(r"\S{50,}", " ", e["text"])
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    pieces = []

    for sentence in sentences:
        words = sentence.split()
        if len(words) <= 50:
            pieces.append(sentence)
        else: 
            for start in range(0, len(words), 50):
                pieces.append(" ".join(words[start:start + 50]))

    for sentence in pieces:
        words = sentence.split()
        if len(current) + len(words) > 25 and len(current) > 0:
            chunks.append(" ".join(current))
            current = []
        current.extend(words)
    
    if len(current) > 0:
        chunks.append((" ".join(current)))

    for parcel in chunks:
        batch.append(stanza.Document([], text=parcel))

    if len(batch) >= 512:
        docs = nlp(batch)
        for doc in docs:
            for sentence in doc.sentences:
                for w in sentence.words: 
                    word = w.text
                    pos = w.upos
                    freq[(word.lower(), pos)] += 1
        batch = []
    # if i > 1000: break

with open("word_frequency_de.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["word", "upos", "count"])
    for (word, pos), count in freq.most_common():
        if pos != "PUNCT" and pos != "SYM" and pos != "X":
            writer.writerow([word, pos, count])