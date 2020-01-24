from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
from multiprocessing import Pool
import re

import numpy
from tqdm import tqdm
from utils import find_transcript_files, transcript_to_list, read_txt, read_tsv, commonvoice_to_list, ami_sdm_to_list, ami_ihm_to_list, ami_mdm_to_list, ted_to_list, remove_punct, check_file
from functools import partial

LOG_STR = " To regenerate this file, please, remove it."

alpha = re.compile(r"^[a-zA-Z\s]+$")


def checkfile(line):
    filename = line.split(" ")[1]
    if not check_file(filename):
        print(f"{filename} is corrupt!")


def check_commonvoice(audio_path, text_path, lists_path, processes):
    for f in ['dev', 'test', 'train']:
        dst_list = os.path.join(lists_path, f"commonvoice-{f}.lst")
        with open(dst_list, "r") as list_f:
            with Pool(processes) as p:
                tqdm(
                    p.imap(checkfile, list_f),
                    total=len(list_f),
                )

    print("Checked CommonVoice", flush=True)


def check_ami_ihm(audio_path, text_path, lists_path, processes):
    for f in ['dev', 'test', 'train']:
        dst_list = os.path.join(lists_path, f"ami-ihm-{f}.lst")
        with open(dst_list, "r") as list_f:
            with Pool(processes) as p:
                tqdm(
                    p.imap(checkfile, list_f),
                    total=len(list_f),
                )


    print("Prepared AMI IHM", flush=True)


def check_ami_sdm(audio_path, text_path, lists_path, processes):
    for f in ['dev', 'test', 'train']:
        dst_list = os.path.join(lists_path, f"ami-sdm-{f}.lst")
        with open(dst_list, "r") as list_f:
            with Pool(processes) as p:
                tqdm(
                    p.imap(checkfile, list_f),
                    total=len(list_f),
                )


    print("Checked AMI SDM1", flush=True)


def check_ami_mdm(audio_path, text_path, lists_path, processes):
    for f in ['dev', 'test', 'train']:
        dst_list = os.path.join(lists_path, f"ami-mdm-{f}.lst")
        with open(dst_list, "r") as list_f:
            with Pool(processes) as p:
                tqdm(
                    p.imap(checkfile, list_f),
                    total=len(list_f),
                )

    print("Checked AMI MDM8", flush=True)


def check_ted(audio_path, text_path, lists_path, processes):
    for f in ['dev', 'test', 'train']:
        dst_list = os.path.join(lists_path, f"ted-{f}.lst")
        with open(dst_list, "r") as list_f:
            with Pool(processes) as p:
                tqdm(
                    p.imap(checkfile, list_f),
                    total=len(list_f),
                )

    print("Checked TED-LIUM", flush=True)


def check_libri(audio_path, text_path, lists_path, processes):
    for f in ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other']:
        dst_list = os.path.join(lists_path, f"ami-mdm-{f}.lst")
        with open(dst_list, "r") as list_f:
            with Pool(processes) as p:
                tqdm(
                    p.imap(checkfile, list_f),
                    total=len(list_f),
                )

    print("Checked LibriSpeech", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined Dataset creation.")
    parser.add_argument(
        "--dst",
        help="destination directory where to store data",
        default="./data_dir",
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

    check_ted(audio_path, text_path, lists_path, args.process)

    check_ami_sdm(audio_path,
                  text_path, lists_path, args.process)
    check_ami_ihm(audio_path,
                  text_path, lists_path, args.process)
    check_ami_mdm(audio_path,
                  text_path, lists_path, args.process)
    check_commonvoice(audio_path,
                      text_path, lists_path, args.process)
    check_libri(audio_path, text_path, lists_path, args.process)
