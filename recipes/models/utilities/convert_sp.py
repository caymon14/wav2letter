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
import sentencepiece_model_pb2 as model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sp Model conversion")

    parser.add_argument(
        "--sp_model", help="sp path",
    )

    parser.add_argument(
        "--tokens", help="sp path",
    )

    args = parser.parse_args()

    m = model.ModelProto()
    m.ParseFromString(open(args.sp_model, 'rb').read())

    vocab = []
    with open(args.tokens, "r") as f:
        for line in f:
            line = line.replace("_", "\u2581").replace("\n", "")
            vocab.append(line)
    print(f'Vocab len {len(vocab)}')
    print(f'Pieces len {len(m.pieces)}')

    for i, p in enumerate(m.pieces):
        if p.score !=0 :
            p.piece = vocab[i-3]


    with open('new.model', 'wb') as f:
        f.write(m.SerializeToString())



    print("Done!", flush=True)
