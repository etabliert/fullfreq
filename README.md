# fullfreq
 word frequency lists with POS tags, using stanza to scan [fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (for now) <br>
 works, but unfinished, will be updated in the future. currently you have to change language on your own in the script config

## requirements
 * python 3.10+, pytorch 2.0+   
 * decent-ish GPU (although you could do use_gpu=False in makeFreq), rtx 2060 needs about 15 minutes for 26M tokens
 * you need a big dataset like fineweb 2 data as jsonl for example (getData.py downloads the deu_Latn of fineweb-2, 50k pages)
 * you need stanza, spacy, tqdm, orjson, torch, pyarrow and datatrove (everything else is python standard library)

## setup
* make sure you have [python](https://www.python.org/) installed <br>
* check the config of each script: default language is de (german), you can change it to any language you want though (you have to change both files)
* run to install:  `` pip install -r requirements.txt `` <br>
    or:  `` pip install stanza spacy tqdm orjson torch pyarrow datatrove`` <br>
* run to get data: `` python getData.py `` (this creates the file fineweb_data_de.jsonl) <br>

## run
* then scan the data: `` python makeFreq.py `` <br> (takes about 15 minutes for me on rtx 2060 for 50k documents, ~26 million tokens)

## output
the output (word_frequency_de.csv) should look something like this (might need to be cleaned up):
```
word,upos,count
der,DET,637044
und,CCONJ,610308
die,DET,587806
in,ADP,523097
dem,DET,442687
von,ADP,253497
...
```
about 1.2 million lines of csv (only cleaning pos: PUNCT, SYM and X)
 
## data / license
* code is licensed with MIT (see LICENSE)
* uses data from [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (deu_Latn, 50k pages): ODC-BY v1.0