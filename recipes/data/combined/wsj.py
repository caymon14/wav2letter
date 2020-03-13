"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to package original WSJ datasets into a form readable in wav2letter++
pipelines

Please install `sph2pipe` on your own -
see https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools \
  with commands :

  wget https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/ctools/sph2pipe_v2.5.tar.gz
  tar -xzf sph2pipe_v2.5.tar.gz && cd sph2pipe_v2.5
  gcc -o sph2pipe *.c -lm

Command : python3 prepare.py --wsj0 [...]/WSJ0/media \
    --wsj1 [...]/WSJ1/media --dst [...] --sph2pipe [...]/sph2pipe_v2.5/sph2pipe

Replace [...] with appropriate paths
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import re
import subprocess
from multiprocessing import Pool

import numpy
from tqdm import tqdm
from wsj_utils import convert_to_flac, find_transcripts, ndx_to_samples, preprocess_word


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSJ Dataset creation.")
    parser.add_argument(
        "--wsj0", help="top level directory containing all WSJ0 discs")
    parser.add_argument("--dst", help="destination directory", default="./wsj")

    parser.add_argument(
        "--sph2pipe",
        help="path to sph2pipe executable",
        default="./sph2pipe_v2.5/sph2pipe",
    )
    parser.add_argument(
        "-p", "--process", help="# of process for Multiprocessing", default=8, type=int
    )

    args = parser.parse_args()

    assert os.path.isdir(str(args.wsj0)), "WSJ0 directory is not found - '{d}'".format(
        d=args.wsj0
    )

    assert os.path.exists(args.sph2pipe), "sph2pipe not found '{d}'".format(
        d=args.sph2pipe
    )

    # Prepare audio data
    transcripts = find_transcripts([args.wsj0])

    subsets = dict()
    subsets["si84"] = ndx_to_samples(
        args.wsj0,
        "11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx",
        transcripts,
        lambda line: None if "11_2_1:wsj0/si_tr_s/401" in line else line,
    )
    assert len(subsets["si84"]
               ) == 7138, "Incorrect number of samples in si84 part: should be 7138, but fould #{}.".format(len(subsets["si84"]))

    subsets["si284"] = ndx_to_samples(
        args.wsj0,
        "11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx",
        transcripts,
        lambda line: None if "11_2_1:wsj0/si_tr_s/401" in line else line,
    )
    assert len(subsets["si284"]
               ) == 37416, "Incorrect number of samples in si284 part: should be 37416, but fould {}.".format(len(subsets["si284"]))

    subsets["nov92"] = ndx_to_samples(
        args.wsj0,
        "11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx",
        transcripts,
        lambda line: line + ".wv1",
    )
    assert (
        len(subsets["nov92"]) == 333
    ), "Incorrect number of samples in nov92 part: should be 333, but fould {}.".format(
        len(subsets["nov92"])
    )

    subsets["nov92_5k"] = ndx_to_samples(
        args.wsj0,
        "11-13.1/wsj0/doc/indices/test/nvp/si_et_05.ndx",
        transcripts,
        lambda line: line + ".wv1",
    )
    assert (
        len(subsets["nov92_5k"]) == 330
    ), "Incorrect number of samples in nov92_5k part: should be 330, but fould {}.".format(
        len(subsets["nov92_5k"])
    )

    audio_path = os.path.join(args.dst, "audio")
    text_path = os.path.join(args.dst, "text")
    lists_path = os.path.join(args.dst, "lists")
    os.makedirs(audio_path, exist_ok=True)
    os.makedirs(text_path, exist_ok=True)
    os.makedirs(lists_path, exist_ok=True)
    transcription_words = set()

    for set_name, samples in subsets.items():
        n_samples = len(samples)
        print(
            "Writing {s} with {n} samples\n".format(s=set_name, n=n_samples), flush=True
        )
        data_dst = os.path.join(audio_path, set_name)
        if os.path.exists(data_dst):
            print(
                """The folder {} exists, existing flac for this folder will be skipped for generation.
                Please remove the folder if you want to regenerate the data""".format(
                    data_dst
                ),
                flush=True,
            )
        with Pool(args.process) as p:
            os.makedirs(data_dst, exist_ok=True)
            samples_info = list(
                tqdm(
                    p.imap(
                        convert_to_flac,
                        zip(
                            samples,
                            numpy.arange(n_samples),
                            [data_dst] * n_samples,
                            [args.sph2pipe] * n_samples,
                        ),
                    ),
                    total=n_samples,
                )
            )
            list_dst = os.path.join(lists_path, set_name + ".lst")
            if not os.path.exists(list_dst):
                with open(list_dst, "w") as f_list:
                    for sample_info in samples_info:
                        f_list.write(" ".join(sample_info) + "\n")
            else:
                print(
                    "List {} already exists, skip its generation."
                    " Please remove it if you want to regenerate the list".format(
                        list_dst
                    ),
                    flush=True,
                )

        for sample_info in samples_info:
            transcription_words.update(sample_info[3].lower().split(" "))
        # Prepare text data
        text_dst = os.path.join(text_path, set_name + ".txt")
        if not os.path.exists(text_dst):
            with open(text_dst, "w") as f_text:
                for sample_info in samples_info:
                    f_text.write(sample_info[3] + "\n")
        else:
            print(
                "Transcript text file {} already exists, skip its generation."
                " Please remove it if you want to regenerate the list".format(
                    text_dst),
                flush=True,
            )

    print("Done!", flush=True)
