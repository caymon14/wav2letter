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

import numpy
from tqdm import tqdm
from utils import find_transcript_files, transcript_to_list, read_txt, convert_to_flac

LOG_STR = " To regenerate this file, please, remove it."


def prepare_commonvoice(commonvoice_location, audio_path, text_path, lists_path):
    print("Prepared CommonVoice", flush=True)

def prepare_ami_ihm(ami_ihm_location, audio_path, text_path, lists_path):
    for f in ['dev', 'test', 'train']:
        dst_list = os.path.join(lists_path, f"ami-ihm-{f}.lst")
        dst_text = os.path.join(text_path, f"ami-ihm-{f}.txt")
        if  not os.path.exists(dst_list):
            with open(dst_list, "w") as list_f,  open(dst_text, "w") as text_f: 
                for name, text in read_txt(os.path.join(ami_ihm_location, f)):
                    _, scenario, headphone, _, start, end = name.split("_")
                    text_f.write(f"{text}\n")
                    export_dir = f"{audio_path}/ihm/{scenario}"
                    lst_record = convert_to_flac(f"{ami_ihm_location}/{scenario}/audio/{scenario}.Headset-{int(headphone[2:])}.wav",
                        int(start)*10, int(end)*10, name, export_dir, text)
                    list_f.write(f"{lst_record}\n")
        else:
            print(f"{dst_list} exists, skipping")

    print("Prepared AMI IHM", flush=True)

def prepare_ami_sdm(ami_sdm_location, audio_path, text_path, lists_path):
    for f in ['dev', 'test', 'train']:
        dst_list = os.path.join(lists_path, f"ami-sdm-{f}.lst")
        dst_text = os.path.join(text_path, f"ami-sdm-{f}.txt")
        if  not os.path.exists(dst_list):
            with open(dst_list, "w") as list_f, open(dst_text, "w") as text_f: 
                for name, text in read_txt(os.path.join(ami_sdm_location, f)):
                    _, scenario, _, _, start, end = name.split("_")
                    text_f.write(f"{text}\n")
                    export_dir = f"{audio_path}/sdm/{scenario}"
                    lst_record = convert_to_flac(f"{ami_sdm_location}/{scenario}/audio/{scenario}.Array1-01.wav",
                            int(start)*10, int(end)*10, name, export_dir, text)
                    list_f.write(f"{lst_record}\n")
        else:
            print(f"{dst_list} exists, skipping")

    print("Prepared AMI SDM1", flush=True)

def prepare_ted(ted_location, audio_path, text_path, lists_path):
    for f in ['dev', 'test', 'train']:
        dst_list = os.path.join(lists_path, f"ted-{f}.lst")
        dst_text = os.path.join(text_path, f"ted-{f}.txt")
        if  not os.path.exists(dst_list):
            with open(dst_list, "w") as list_f, open(dst_text, "w") as text_f: 
                for name, text in read_txt(os.path.join(ted_location, f)):
                    scenario, start, end = name.split("-")
                    text_f.write(f"{text}\n")
                    export_dir = f"{audio_path}/ted/{scenario}"
                    lst_record = convert_to_flac(f"{ted_location}/legacy/{f}/sph/{scenario}.sph",
                            int(start)*10, int(end)*10, name, export_dir, text)
                    list_f.write(f"{lst_record}\n")
        else:
            print(f"{dst_list} exists, skipping")

    print("Prepared TED-LIUM", flush=True)

def prepare_libri(libri_location, audio_path, text_path, lists_path, processes):
    subpaths = {
        "train": ["train-clean-100"],
        "dev": ["dev-other"],
        "test": ["test-other"],
    }

    subpath_names = numpy.concatenate(list(subpaths.values()))

    # Prepare the audio data
    print("Converting audio data into necessary format.", flush=True)
    word_dict = {}
    for subpath_type in subpaths.keys():
        word_dict[subpath_type] = set()
        for subpath in subpaths[subpath_type]:
            src = os.path.join(libri_location, subpath)
            assert os.path.exists(src), "Unable to find the directory - '{src}'".format(
                src=src
            )

            dst_list = os.path.join(lists_path, subpath + ".lst")
            if os.path.exists(dst_list):
                print(
                    "Path {} exists, skip its generation.".format(dst_list) + LOG_STR,
                    flush=True,
                )
                continue

            print("Analyzing {src}...".format(src=src), flush=True)
            transcript_files = find_transcript_files(src)
            transcript_files.sort()

            print("Writing to {dst}...".format(dst=dst_list), flush=True)
            with Pool(processes) as p:
                samples = list(
                    tqdm(
                        p.imap(transcript_to_list, transcript_files),
                        total=len(transcript_files),
                    )
                )

            with open(dst_list, "w") as fout:
                for sp in samples:
                    for s in sp:
                        word_dict[subpath_type].update(s[-1].split(" "))
                        s[0] = subpath + "-" + s[0]
                        fout.write(" ".join(s) + "\n")

    for pname in subpath_names:
        current_path = os.path.join(text_path, pname + ".txt")
        if not os.path.exists(current_path):
            with open(os.path.join(lists_path, pname + ".lst"), "r") as flist, open(
                os.path.join(text_path, pname + ".txt"), "w"
            ) as fout:
                for line in flist:
                    fout.write(" ".join(line.strip().split(" ")[3:]) + "\n")
        else:
            print(
                "Path {} exists, skip its generation.".format(current_path) + LOG_STR,
                flush=True,
            )

    print("Prepared LibriSpeech", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined Dataset creation.")
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

    args = parser.parse_args()

    audio_path = os.path.join(args.dst, "audio")
    text_path = os.path.join(args.dst, "text")
    lists_path = os.path.join(args.dst, "lists")
    os.makedirs(audio_path, exist_ok=True)
    os.makedirs(text_path, exist_ok=True)
    os.makedirs(lists_path, exist_ok=True)  

    prepare_libri(args.libri, audio_path, text_path, lists_path, args.process)

    prepare_commonvoice(args.commonvoice, audio_path, text_path, lists_path)
    
    prepare_ted(args.ted, audio_path, text_path, lists_path)
    prepare_ami_sdm(args.ami_sdm, audio_path, text_path, lists_path)
    prepare_ami_ihm(args.ami_ihm, audio_path, text_path, lists_path)


