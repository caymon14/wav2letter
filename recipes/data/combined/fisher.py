from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
from multiprocessing import Pool
import re

import numpy
from tqdm import tqdm
from utils import remove_punct
from functools import partial
from glob import glob
from pydub import AudioSegment

alpha = re.compile(r"^[a-zA-Z\s]+$") 
max_duration = 10 * 1000
test_every = 5

def cure_text(text):
    text = remove_punct(text)
    text = text.lower()
    return " ".join(map(lambda x: x.strip(), text.split(" ")))

def read_dialogs(lines):
    dialogs = []
    for line in lines:
        if len(line) > 0 and not line.startswith("#"):  
            chunks = line.split()
            text = chunks[3:]
            start_seconds = float(chunks[0]) * 1000
            end_seconds = float(chunks[1]) * 1000
            start = float(start_seconds)
            end = float(end_seconds)
            dialogs.append((text, start, end))

    return dialogs
            

def prepare_fisher(fisher, audio_path, text_path, lists_path, processes, sph2pipe):
    train_file = f"{lists_path}/fisher-train.lst"
    test_file = f"{lists_path}/fisher-test.lst"
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        with open(train_file, "w") as lst, open(test_file, "w") as lst_test:
            for file in glob(f"{fisher}/*.txt"):
                with Pool(args.process) as p:
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

    else:
        print(f"{train_file} exists, doing verify")
        new_list = []
        with open(train_file, "r") as list_f:
            for line in list_f:
                filename = line.split(" ")[1]
                text = " ".join(line.strip().split(" ")[3:])
                text = re.sub(' +', ' ', text)
                params = " ".join(line.strip().split(" ")[:3])
                line = f"{params} {text}\n"
                if not os.path.exists(filename) or len(text) < 2 or not alpha.match(text):
                    print(f"{filename} does not exists or text is empty, text: {text}")
                else:
                    new_list.append(line)
        with open(train_file, "w") as list_f:
            list_f.writelines(new_list)
    
    print("Prepared fisher", flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Combined Dataset creation.")
    parser.add_argument(
        "--dst",
        help="destination directory where to store data",
        default="./data_dir",
    )
    parser.add_argument(
        "--fisher",
        help="Fisher data location",
        default="./fisher",
    )

    parser.add_argument(
        "-p",
        "--process",
        help="number of process for multiprocessing",
        default=8,
        type=int,
    )

    parser.add_argument(
        "--sph2pipe",
        help="path to sph2pipe executable",
        default="./sph2pipe_v2.5/sph2pipe",
    )

    args=parser.parse_args()

    audio_path=os.path.join(args.dst, "audio")
    text_path=os.path.join(args.dst, "text")
    lists_path=os.path.join(args.dst, "lists")
    os.makedirs(f"{audio_path}/fisher", exist_ok=True)
    os.makedirs(text_path, exist_ok=True)
    os.makedirs(lists_path, exist_ok=True)

    prepare_fisher(args.fisher, audio_path, text_path, lists_path, args.process, args.sph2pipe)

