import stanza
import spacy
import json
from tqdm import tqdm
import csv
from collections import Counter
import re

dataset = "fineweb_data_de.jsonl"
nlp = stanza.Pipeline("de", processors="tokenize,pos", tokenize_pretokenized=True, use_gpu=True, pos_batch_size=4000, verbose=False)
nlp_token = spacy.blank("de")
length = sum(1 for l in open(dataset, encoding="utf-8"))

freq = Counter()
batch = []

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
            for sentence in doc.sentences:
                for w in sentence.words: 
                    word = w.text
                    pos = w.upos
                    freq[(word.lower(), pos)] += 1
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

with open("word_frequency_de.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["word", "upos", "count"])
    for (word, pos), count in freq.most_common():
        if pos != "PUNCT" and pos != "SYM" and pos != "X":
            writer.writerow([word, pos, count])