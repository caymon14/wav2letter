"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

----------

Script to package original Librispeech datasets into a form readable in
wav2letter++ pipelines

Command : python3 prepare.py --dst [...]

Replace [...] with appropriate path
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
from multiprocessing import Pool
import re

import numpy
from tqdm import tqdm
from utils import find_transcript_files, transcript_to_list, read_txt, read_tsv, commonvoice_to_list, ami_sdm_to_list, ami_ihm_to_list, ami_mdm_to_list, ted_to_list, remove_punct
from functools import partial

LOG_STR = " To regenerate this file, please, remove it."



def prepare_commonvoice(commonvoice_location, audio_path, text_path, lists_path, processes):
    for f in ['dev', 'test', 'train']:
        dst_list = os.path.join(lists_path, f"commonvoice-{f}.lst")
        dst_text = os.path.join(text_path, f"commonvoice-{f}.txt")
        if not os.path.exists(dst_list):
            to_list = partial(commonvoice_to_list, audio_path, f, commonvoice_location)
            with Pool(processes) as p:
                rows = read_tsv(os.path.join(commonvoice_location, f"{f}.tsv"))
                samples = list(
                    tqdm(
                        p.imap(to_list, rows),
                        total=len(rows),
                    )
                )
            with open(dst_list, "w") as list_f:
                list_f.writelines(samples)
            
            with open(dst_list, "r") as list_f,  open(dst_text, "w") as text_f:
                for line in list_f:
                    text_f.write(" ".join(line.strip().split(" ")[3:]) + "\n")

        else:
            print(f"{dst_list} exists, doing verify")
            new_list = []
            with open(dst_list, "r") as list_f:
                for line in list_f:
                    filename = line.split(" ")[1]
                    text = " ".join(line.strip().split(" ")[3:])
                    params = " ".join(line.strip().split(" ")[:3])
                    text = remove_punct(text)
                    line = f"{params} {text}\n"
                    if not os.path.exists(filename) or len(text) < 2 or not text.isalpha():
                        print(f"{filename} does not exists or text is empty, text: {text}")
                    else:
                        new_list.append(line)
            with open(dst_list, "w") as list_f:
                list_f.writelines(new_list)

    print("Prepared CommonVoice", flush=True)


def prepare_ami_ihm(ami_ihm_location, audio_path, text_path, lists_path, processes):
    for f in ['dev', 'test', 'train']:
        dst_list=os.path.join(lists_path, f"ami-ihm-{f}.lst")
        dst_text=os.path.join(text_path, f"ami-ihm-{f}.txt")
        if not os.path.exists(dst_list):
            with Pool(processes) as p:
                to_list = partial(ami_ihm_to_list, audio_path, ami_ihm_location)
                rows = read_txt(os.path.join(ami_ihm_location, f))
                samples = list(
                    tqdm(
                        p.imap(to_list, rows),
                        total=len(rows),
                    )
                )
            with open(dst_list, "w") as list_f:
                list_f.writelines(samples)
            
            with open(dst_list, "r") as list_f,  open(dst_text, "w") as text_f:
                for line in list_f:
                    text_f.write(" ".join(line.strip().split(" ")[3:]) + "\n")
        else:
            print(f"{dst_list} exists, doing verify")
            new_list = []
            with open(dst_list, "r") as list_f:
                for line in list_f:
                    filename = line.split(" ")[1]
                    text = " ".join(line.strip().split(" ")[3:])
                    params = " ".join(line.strip().split(" ")[:3])
                    text = remove_punct(text)
                    line = f"{params} {text}\n"
                    if not os.path.exists(filename) or len(text) < 2 or not text.isalpha():
                        print(f"{filename} does not exists or text is empty, text: {text}")
                    else:
                        new_list.append(line)
            with open(dst_list, "w") as list_f:
                list_f.writelines(new_list)


    print("Prepared AMI IHM", flush=True)


def prepare_ami_sdm(ami_sdm_location, audio_path, text_path, lists_path, processes):
    for f in ['dev', 'test', 'train']:
        dst_list=os.path.join(lists_path, f"ami-sdm-{f}.lst")
        dst_text=os.path.join(text_path, f"ami-sdm-{f}.txt")
        if not os.path.exists(dst_list):
            with Pool(processes) as p:
                to_list = partial(ami_sdm_to_list, audio_path, ami_sdm_location)
                rows = read_txt(os.path.join(ami_sdm_location, f))
                samples = list(
                    tqdm(
                        p.imap(to_list, rows),
                        total=len(rows),
                    )
                )
            with open(dst_list, "w") as list_f:
                list_f.writelines(samples)
            
            with open(dst_list, "r") as list_f,  open(dst_text, "w") as text_f:
                for line in list_f:
                    text_f.write(" ".join(line.strip().split(" ")[3:]) + "\n")
                    
        else:
            print(f"{dst_list} exists, doing verify")
            new_list = []
            with open(dst_list, "r") as list_f:
                for line in list_f:
                    filename = line.split(" ")[1]
                    text = " ".join(line.strip().split(" ")[3:])
                    params = " ".join(line.strip().split(" ")[:3])
                    text = remove_punct(text)
                    line = f"{params} {text}\n"
                    if not os.path.exists(filename) or len(text) < 2 or not text.isalpha():
                        print(f"{filename} does not exists or text is empty, text: {text}")
                    else:
                        new_list.append(line)
            with open(dst_list, "w") as list_f:
                list_f.writelines(new_list)


    print("Prepared AMI SDM1", flush=True)


def prepare_ami_mdm(ami_mdm_location, audio_path, text_path, lists_path, processes):
    for f in ['dev', 'test', 'train']:
        dst_list=os.path.join(lists_path, f"ami-mdm-{f}.lst")
        dst_text=os.path.join(text_path, f"ami-mdm-{f}.txt")
        if not os.path.exists(dst_list):
            with Pool(processes) as p:
                to_list = partial(ami_mdm_to_list, audio_path, ami_mdm_location)
                rows = read_txt(os.path.join(ami_mdm_location, f))
                samples = list(
                    tqdm(
                        p.imap(to_list, rows),
                        total=len(rows),
                    )
                )
            with open(dst_list, "w") as list_f:
                list_f.writelines(samples)
            
            with open(dst_list, "r") as list_f,  open(dst_text, "w") as text_f:
                for line in list_f:
                    text_f.write(" ".join(line.strip().split(" ")[3:]) + "\n")

        else:
            print(f"{dst_list} exists, doing verify")
            new_list = []
            with open(dst_list, "r") as list_f:
                for line in list_f:
                    filename = line.split(" ")[1]
                    text = " ".join(line.strip().split(" ")[3:])
                    params = " ".join(line.strip().split(" ")[:3])
                    text = remove_punct(text)
                    line = f"{params} {text}\n"
                    if not os.path.exists(filename) or len(text) < 2 or not text.isalpha():
                        print(f"{filename} does not exists or text is empty, text: {text}")
                    else:
                        new_list.append(line)
            with open(dst_list, "w") as list_f:
                list_f.writelines(new_list)


    print("Prepared AMI MDM8", flush=True)


def prepare_ted(ted_location, audio_path, text_path, lists_path, processes):
    for f in ['dev', 'test', 'train']:
        dst_list=os.path.join(lists_path, f"ted-{f}.lst")
        dst_text=os.path.join(text_path, f"ted-{f}.txt")
        if not os.path.exists(dst_list):
            with Pool(processes) as p:
                to_list = partial(ted_to_list, audio_path, f, ted_location)
                rows = read_txt(os.path.join(ted_location, f))
                samples = list(
                    tqdm(
                        p.imap(to_list, rows),
                        total=len(rows),
                    )
                )
            with open(dst_list, "w") as list_f:
                list_f.writelines(samples)
            
            with open(dst_list, "r") as list_f,  open(dst_text, "w") as text_f:
                for line in list_f:
                    text_f.write(" ".join(line.strip().split(" ")[3:]) + "\n")

        else:
            print(f"{dst_list} exists, doing verify")
            new_list = []
            with open(dst_list, "r") as list_f:
                for line in list_f:
                    filename = line.split(" ")[1]
                    text = " ".join(line.strip().split(" ")[3:])
                    params = " ".join(line.strip().split(" ")[:3])
                    text = remove_punct(text)
                    line = f"{params} {text}\n"
                    if not os.path.exists(filename) or len(text) < 2 or not text.isalpha():
                        print(f"{filename} does not exists or text is empty, text: {text}")
                    else:
                        new_list.append(line)
            with open(dst_list, "w") as list_f:
                list_f.writelines(new_list)
    print("Prepared TED-LIUM", flush=True)


def prepare_libri(libri_location, audio_path, text_path, lists_path, processes):
    subpaths={
        "train": ["train-clean-100", "train-clean-360", "train-other-500"],
        "dev": ["dev-clean", "dev-other"],
        "test": ["test-clean", "test-other"],
    }

    subpath_names=numpy.concatenate(list(subpaths.values()))

    # Prepare the audio data
    print("Converting audio data into necessary format.", flush=True)
    word_dict={}
    for subpath_type in subpaths.keys():
        word_dict[subpath_type]=set()
        for subpath in subpaths[subpath_type]:
            src=os.path.join(libri_location, subpath)
            assert os.path.exists(src), "Unable to find the directory - '{src}'".format(
                src=src
            )

            dst_list=os.path.join(lists_path, subpath + ".lst")
            if os.path.exists(dst_list):
                print(
                    "Path {} exists, skip its generation.".format(
                        dst_list) + LOG_STR,
                    flush=True,
                )
                continue

            print("Analyzing {src}...".format(src=src), flush=True)
            transcript_files=find_transcript_files(src)
            transcript_files.sort()

            print("Writing to {dst}...".format(dst=dst_list), flush=True)
            with Pool(processes) as p:
                samples=list(
                    tqdm(
                        p.imap(transcript_to_list, transcript_files),
                        total=len(transcript_files),
                    )
                )

            with open(dst_list, "w") as fout:
                for sp in samples:
                    for s in sp:
                        word_dict[subpath_type].update(s[-1].split(" "))
                        s[0]=subpath + "-" + s[0]
                        fout.write(" ".join(s) + "\n")

    for pname in subpath_names:
        current_path=os.path.join(text_path, pname + ".txt")
        if not os.path.exists(current_path):
            with open(os.path.join(lists_path, pname + ".lst"), "r") as flist, open(
                os.path.join(text_path, pname + ".txt"), "w"
            ) as fout:
                for line in flist:
                    fout.write(" ".join(line.strip().split(" ")[3:]) + "\n")
        else:
            print(
                "Path {} exists, skip its generation.".format(
                    current_path) + LOG_STR,
                flush=True,
            )

    print("Prepared LibriSpeech", flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Combined Dataset creation.")
    parser.add_argument(
        "--dst",
        help="destination directory where to store data",
        default="./data_dir",
    )
    parser.add_argument(
        "--libri",
        help="Libri data location",
        default="./LibriSpeech",
    )
    parser.add_argument(
        "--commonvoice",
        help="CommonVoice data location",
        default="./commonvoice",
    )
    parser.add_argument(
        "--ami_ihm",
        help="AMI IHM data location",
        default="./ihm",
    )
    parser.add_argument(
        "--ami_sdm",
        help="AMI SDM data location",
        default="./sdm1",
    )
    parser.add_argument(
        "--ami_mdm",
        help="AMI MDM data location",
        default="./mdm8",
    )
    parser.add_argument(
        "--ted",
        help="TED-LIUM data location",
        default="./TEDLIUM",
    )
    parser.add_argument(
        "-p",
        "--process",
        help="number of process for multiprocessing",
        default=8,
        type=int,
    )

    args=parser.parse_args()

    audio_path=os.path.join(args.dst, "audio")
    text_path=os.path.join(args.dst, "text")
    lists_path=os.path.join(args.dst, "lists")
    os.makedirs(audio_path, exist_ok=True)
    os.makedirs(text_path, exist_ok=True)
    os.makedirs(lists_path, exist_ok=True)



    prepare_ted(args.ted, audio_path, text_path, lists_path, args.process)

    prepare_ami_sdm(args.ami_sdm, audio_path,
                    text_path, lists_path, args.process)
    prepare_ami_ihm(args.ami_ihm, audio_path,
                    text_path, lists_path, args.process)
    prepare_ami_mdm(args.ami_mdm, audio_path,
                    text_path, lists_path, args.process)
    prepare_commonvoice(args.commonvoice, audio_path,
                        text_path, lists_path, args.process)
    prepare_libri(args.libri, audio_path, text_path, lists_path, args.process)
