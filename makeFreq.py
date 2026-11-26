import stanza
import spacy
import orjson
from tqdm import tqdm
import csv
from collections import Counter
import re
import time
import torch
import queue
import threading
torch.backends.cudnn.benchmark = True

# --- CONFIG ---
lang = "de"
pos_batch_size = 32768
doc_batch = 512
# --- config end ---

dataset = f"fineweb_data_{lang}.jsonl"
nlp = stanza.Pipeline(lang, package="default_fast", processors="tokenize,pos", tokenize_pretokenized=True, use_gpu=True, pos_batch_size=pos_batch_size, verbose=False)
nlp_token = spacy.blank(lang)

length = sum(1 for l in open(dataset, encoding="utf-8"))
progressBar = tqdm(total=length, desc="Tagging progress:", position=1)

freq = Counter()
batch = []
total_tokens = 0
total_docs = 0
old_time = time.time()
old_docs = 0
old_tokens = 0
SENTINEL = object()

q = queue.Queue(maxsize=4)

def report():
    global total_tokens, total_docs, old_time, old_docs, old_tokens
    deltaTime = time.time() - old_time
    deltaTokens = total_tokens - old_tokens
    deltaDocs = total_docs - old_docs
    tqdm.write(f"Processed docs: {total_docs}, tokens: {total_tokens} | {round(deltaDocs/deltaTime, 1)} docs/s, {round(deltaTokens/deltaTime, 1)} tokens/s")
    old_time = time.time()
    old_docs = total_docs
    old_tokens = total_tokens

def reading():
    global batch
    try:
        for line in tqdm(open(dataset, encoding="utf-8"), total=length, desc="Reading progress:", position=0):
            e = orjson.loads(line)
            text = re.sub(r"\S{50,}", " ", e["text"])
            sentences = re.split(r'(?<=[.!?])\s+', text)
            pieces = []
            pack = []

            for sentence in sentences:      # kinda disregards sentences and puts every token into a pack of 100 tokens. improves speed by a lot
                if not sentence.strip():
                    continue
                tokenized = nlp_token(sentence)
                tSentence = [token.text for token in tokenized]
                for token in tSentence:
                    pieces.append(token)
                    if len(pieces) >= 100:
                        pack.append(pieces)
                        pieces = []
            if pieces: pack.append(pieces)

            batch.append(stanza.Document([], text=pack))
            if len(batch) >= doc_batch:
                q.put(batch)
                batch = []

        if batch:
            q.put(batch)
    finally:
        q.put(SENTINEL)

def tagging():
    global total_docs, total_tokens
    while True:
        current_batch = q.get()
        if current_batch is SENTINEL: break
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            docs = nlp(current_batch)
        for doc in docs:
            total_docs += 1
            progressBar.update(1)
            for sentence in doc.sentences:
                for w in sentence.words: 
                    total_tokens += 1
                    word = w.text
                    pos = w.upos
                    freq[(word.lower(), pos)] += 1
        report()
    progressBar.close()

def main():
    print(f"Started processing {lang}: {dataset}, {length} documents")
    threading.Thread(target=reading, daemon=True).start()
    tagger = threading.Thread(target=tagging, daemon=True)
    tagger.start()
    

    tagger.join()
    with open(f"word_frequency_{lang}.csv", "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "upos", "count"])
        for (word, pos), count in freq.most_common():
            if pos != "PUNCT" and pos != "SYM" and pos != "X":
                writer.writerow([word, pos, count])

if __name__ == "__main__":
    main()