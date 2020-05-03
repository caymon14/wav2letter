"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to prepare recipe to train/eval model on Librispeech in wav2letter++ pipelines

Command : python3 prepare.py --data_dst [...] --model_dst [...] --wp 10000 --nbest 10

Replace [...] with appropriate paths
"""

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
        "--lm_path", help="lm location",
    )

    parser.add_argument(
        "--sp_model", help="sp path",
    )

    parser.add_argument(
        "--out_path", help="out path",
    )
 
    parser.add_argument(
        "--nbest",
        help="number of best segmentations for each word (or numbers comma separated)",
        default="10",
    )

    args = parser.parse_args()

    lm_words = []
    with open(args.lm_path, "r") as arpa:
        for line in arpa:
            # verify if the line corresponds to unigram
            if not re.match(r"[-]*[0-9\.]+\t\S+\t*[-]*[0-9\.]*$", line):
                continue
            word = line.split("\t")[1]
            word = word.strip().lower()
            if word == "<unk>" or word == "<s>" or word == "</s>":
                continue
            assert re.match(
                "^[a-z']+$", word), "invalid word - {w}".format(w=word)
            lm_words.append(word)

    # word -> word piece lexicon for loading targets
    print("Creating word -> word pieces lexicon...\n", flush=True)
    sp = spm.SentencePieceProcessor()
    sp.Load(args.sp_model)

    for nbest in args.nbest.split(","):
        nbest = int(nbest)
        decoder_lexicon_name = "lexicon.txt"
        with open(os.path.join(args.out_path, decoder_lexicon_name), "w") as f_lexicon:
            for word in lm_words:
                wps = sp.NBestEncodeAsPieces(word, nbest)
                for wp in wps:  # the order matters for our training
                    f_lexicon.write(
                        word
                        + "\t"
                        + " ".join([w.replace("\u2581", "_") for w in wp])
                        + "\n"
                    )
    print("Done!", flush=True)
