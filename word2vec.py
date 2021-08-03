# -*- coding: utf-8 -*-

import codecs
import sys

import gensim
from tqdm import tqdm


class Sentences(object):
    def __init__(self, filename: str):
        self.filename = filename

    def __iter__(self):
        for line in tqdm(codecs.open(self.filename, "r", encoding="utf-8"), self.filename):
            yield line.strip().split()


def main(path):
    sentences = Sentences(path)
    model = gensim.models.Word2Vec(sentences, vector_size=200, window=5, min_count=5, negative=5, max_vocab_size=20000)
    model.save("word_vectors/listings.w2v")
    # model.wv.save_word2vec_format("word_vectors/" + domain + ".txt", binary=False)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "preprocessed_data/listings.txt"

    try:
        import os
        os.mkdir("word_vectors/")
    except:
        pass

    print("Training w2v on dataset", path)

    main(path)

    print("Training done.")

    model = gensim.models.Word2Vec.load("word_vectors/listings.w2v")

    for word in ["daybed", "balcony", "houseplant", "elegant", "adventure", "bookstore", "asked", "terminal", "cookware"]:
        if word in model.wv.key_to_index:
            print(word, [w for w, c in model.wv.similar_by_word(word=word)])
        else:
            print(word, "not in vocab")
