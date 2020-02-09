
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import re
from collections import defaultdict

import sentencepiece as spm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Librispeech Dataset creation.")
    parser.add_argument(
        "--model_file", help="data destination directory", default="./data_dir"
    )

    parser.add_argument(
        "--tokens_file",
        help="model auxilary files destination directory",
        default="./model",
    )

    parser.add_argument(
        "--output_folder",
        help="model auxilary files destination directory",
        default="./model",
    )

    args = parser.parse_args()
    sp = spm.SentencePieceProcessor()
    sp.Load(args.model_name)

    vocab = []
    with open(args.tokens_file, "r") as f:
        for i, line in enumerate(f):
            score = i/1000
            line = line..replace("_", "\u2581")
            vocab.append(f"{line}\t-{score}\n")

    with open(f"{args.output_folder}/vocab.lst", "w") as f:
        f.writelines(vocab)

    sp.SetVocabulary(vocab)

    am_path = args.output_folder

    for nbest in "1":
        nbest = int(nbest)
        lexicon_name = "combined-train+dev-unigram-{sz}-nbest{n}.lexicon".format(
            sz=5000, n=nbest
        )

        with open(os.path.join(am_path, lexicon_name), "w") as f_lexicon:
            for word in lexicon_words:
                wps = sp.NBestEncodeAsPieces(word, nbest)
                for wp in wps:  # the order matters for our training
                    f_lexicon.write(
                        word
                        + "\t"
                        + " ".join([w.replace("\u2581", "_") for w in wp])
                        + "\n"
                    )
