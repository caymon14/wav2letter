
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
        "--data_dst", help="data destination directory", default="./data_dir"
    )
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

    subpaths = {
        "train": ["train-clean-100", "train-clean-360", "train-other-500", "ami-sdm-train", "ami-mdm-train", "ami-ihm-train", "ted-train", "commonvoice-train", "callhome-train", "dev-clean", "dev-other", "ami-sdm-dev", "ami-mdm-dev", "ami-ihm-dev", "ted-dev", "commonvoice-dev"],
        "dev": ["test-clean", "test-other", "ami-sdm-test", "ami-mdm-test", "ami-ihm-test", "ted-test", "commonvoice-test", "callhome-test"],
    }

    lists_path = os.path.join(args.data_dst, "lists")

    # Generating am/*
    train_all_text = os.path.join(args.output_folder, "train.txt")

    # prepare data
    print("Preparing tokens and lexicon for acoustic model...\n", flush=True)
    word_dict = defaultdict(set)
    with open(train_all_text, "w") as ftext:
        for key, names in subpaths.items():
            for name in names:
                with open(os.path.join(lists_path, name + ".lst"), "r") as flist:
                    for line in flist:
                        transcription = line.strip().split(" ")[3:]
                        if key == "train":
                            ftext.write(" ".join(transcription) + "\n")
                        word_dict[key].update(transcription)

    lexicon_words = sorted(word_dict["train"] | word_dict["dev"])

    sp = spm.SentencePieceProcessor()
    sp.Load(args.model_file)

    vocab = []
    with open(args.tokens_file, "r") as f:
        for line in f:
            line = line.replace("_", "\u2581").replace("\n", "")
            vocab.append(line)

    with open(f"{args.output_folder}/vocab.lst", "w") as f:
        for i, w in enumerate(vocab):
            score = i/1000
            f.write(f"{w}\t-{score}\n")

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
