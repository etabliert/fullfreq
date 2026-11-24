import stanza
import spacy
import json
from tqdm import tqdm
import csv
from collections import Counter
import re
import time

# config
lang = "de"
batch_size = 4000

dataset = f"fineweb_data_{lang}.jsonl"
nlp = stanza.Pipeline("de", package="default_fast", processors="tokenize,pos", tokenize_pretokenized=True, use_gpu=True, pos_batch_size=4000, verbose=False)
nlp_token = spacy.blank("de")
length = sum(1 for l in open(dataset, encoding="utf-8"))

freq = Counter()
batch = []
total_tokens = 0
total_docs = 0
old_time = time.time()
old_docs = 0
old_tokens = 0

def report():
    global total_tokens, total_docs, old_time, old_docs, old_tokens
    deltaTime = time.time() - old_time
    deltaTokens = total_tokens - old_tokens
    deltaDocs = total_docs - old_docs
    tqdm.write(f"Processed docs: {total_docs} Processed tokens: {total_tokens} | {round(deltaDocs/deltaTime, 1)} docs/s {round(deltaTokens/deltaTime, 1)} tokens/s")
    old_time = time.time()
    old_docs = total_docs
    old_tokens = total_tokens

for i, line in enumerate(tqdm(open(dataset, encoding="utf-8"), total=length)):
    e = json.loads(line)
    text = re.sub(r"\S{50,}", " ", e["text"])
    sentences = re.split(r'(?<=[.!?])\s+', text)
    pieces = []

    for sentence in sentences:
        if not sentence.strip():
            continue
        tokenized = nlp_token(sentence)
        tSentence = [token.text for token in tokenized]
        if len(tSentence) <= 100:
            pieces.append(tSentence)
        else: 
            for start in range(0, len(tSentence), 50):
                pieces.append(tSentence[start:start + 50])

   
    batch.append(stanza.Document([], text=pieces))

    if len(batch) >= 512:   # try other numbers
        docs = nlp(batch)
        for doc in docs:
            total_docs += 1
            for sentence in doc.sentences:
                for w in sentence.words: 
                    total_tokens += 1
                    word = w.text
                    pos = w.upos
                    freq[(word.lower(), pos)] += 1
        report()
        batch = []
    #if i > 5000: break
    
if batch:
    docs = nlp(batch)
    for doc in docs:
        for sentence in doc.sentences:
            for w in sentence.words: 
                word = w.text
                pos = w.upos
                freq[(word.lower(), pos)] += 1
report()

with open(f"word_frequency_{lang}.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["word", "upos", "count"])
    for (word, pos), count in freq.most_common():
        if pos != "PUNCT" and pos != "SYM" and pos != "X":
            writer.writerow([word, pos, count])