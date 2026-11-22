# fullfreq
 word frequency lists with POS tags, using stanza to scan [fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (for now) <br>
 **very unfinished**, will be updated in the future, currently only german

## requirements
 * python 3.10+   
 * decent-ish GPU (although you could do use_gpu=False in makeFreq)
 * you need a big dataset like fineweb 2 data as jsonl for example (getData_de.py downloads the deu_Latn of fineweb-2, 50k pages)
 * you need stanza, tqdm (everything else is python standart library)

## setup
* make sure you have [python](https://www.python.org/) installed <br>
* run to install:     `` pip install stanza tqdm `` <br>
* run to get data: `` python getData_de.py `` (this creates the file fineweb_data_de.jsonl) <br>

## run
* then scan the data: `` python makeFreq_de.py `` <br> (this might take quite a while, about 40-60 minutes for me on rtx 2060 for 50k pages)

## output
the output (word_frequency_de.csv) should look something like this:
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
 
## data / license
* [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (deu_Latn, 50k pages): ODC-BY v1.0