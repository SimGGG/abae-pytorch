# ABAE-PyTorch by SimG

* Update details
  - applying the changed gensim method    :   2021-08-03


Yet another PyTorch implementation of the model described in the paper [**An Unsupervised Neural Attention Model for Aspect Extraction**](https://aclweb.org/anthology/papers/P/P17/P17-1036/) by He, Ruidan and  Lee, Wee Sun  and  Ng, Hwee Tou  and  Dahlmeier, Daniel, **ACL2017**.


Step 1. Data download & Prerprocessing

```
$ wget http://data.insideairbnb.com/united-states/ny/new-york-city/2021-05-06/data/listings.csv.gz # Airbnb listings dataset (2021-05-06)
$ gzip -d listings.csv.gz
$ python preprocess.py
```

Then we need to train the word vectors:
```
python word2vec.py preprocessed_data/listings.txt
```
And run 

```
usage: main.py [-h] [--word-vectors-path <str>] [--batch-size BATCH_SIZE]
               [--aspects-number ASPECTS_NUMBER] [--ortho-reg ORTHO_REG]
               [--epochs EPOCHS] [--optimizer {adam,adagrad,sgd}]
               [--negative-samples NEG_SAMPLES] [--dataset-path DATASET_PATH]
               [--maxlen MAXLEN]

optional arguments:
  -h, --help            show this help message and exit
  --word-vectors-path <str>, -wv <str>
                        path to word vectors file
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Batch size for training
  --aspects-number ASPECTS_NUMBER, -as ASPECTS_NUMBER
                        A total number of aspects
  --ortho-reg ORTHO_REG, -orth ORTHO_REG
                        Ortho-regularization impact coefficient
  --epochs EPOCHS, -e EPOCHS
                        Epochs count
  --optimizer {adam,adagrad,sgd}, -opt {adam,adagrad,sgd}
                        Optimizer
  --negative-samples NEG_SAMPLES, -ns NEG_SAMPLES
                        Negative samples per positive one
  --dataset-path DATASET_PATH, -d DATASET_PATH
                        Path to a training texts file. One sentence per line,
                        tokens separated wiht spaces.
  --maxlen MAXLEN, -l MAXLEN
                        Max length of the considered sentence; the rest is
                        clipped if longer

```

## Thanks for contributions
[@alexdevmotion](https://github.com/alexdevmotion)

## TODOs

* Evaluation: PMI, NPMI, LCP, L1/L2/coord/cosine (Nikolenko SIGIR'16), ...
* Aspects prediction on text + visualization
* Saving the model, aspects, etc.


## Please also see

* Reference [Keras implementation](https://github.com/ruidan/Unsupervised-Aspect-Extraction)
* Updated [Keras version](https://github.com/madrugado/Attention-Based-Aspect-Extraction), compliant with Keras 2.x and Python 3.x.
