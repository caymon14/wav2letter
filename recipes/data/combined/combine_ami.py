from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
from multiprocessing import Pool
import re
import random

import numpy
from tqdm import tqdm
from utils import remove_punct
from functools import partial
from glob import glob
from pydub import AudioSegment

max_duration = 10 * 1000

def to_list(chunk):
    text = []
    for line in chunk:
        filename = line.split(" ")[1]
        t = " ".join(line.strip().split(" ")[3:])

        if len(text) != 0:
            silence = AudioSegment.silent(500, 16000)
            segment = segment.append(silence, crossfade=50)

            next_chunk = AudioSegment.from_file(filename)
            segment = segment.append(next_chunk, crossfade=50)                    
        else:
            segment = AudioSegment.from_file(filename)
        text.append(t.strip())
    file_id = line.split(" ")[0]
    audio_path = "/".join(filename.split("/")[:-3])
    filename = f"{audio_path}/ami_combined/{file_id}.flac"
    segment = segment.set_sample_width(2)
    segment = segment.set_frame_rate(16000)
    segment.export(filename, format="flac")
    dur = len(segment)
    text = " ".join(text).strip()
    
    return f"{file_id} {filename} {dur}.0 {text}\n"
           

def prepare_ami(audio_path, text_path, lists_path, processes):
    train_file = f"{lists_path}/ami-combined-train.lst"
    test_file = f"{lists_path}/ami-combined-test.lst"

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        train = []
        test = []
        for f in ['dev', 'train']:
            for d in ['mdm', 'ihm', 'sdm']:
                with open(os.path.join(lists_path, f"ami-{d}-{f}.lst")) as lst:
                    for line in lst:
                        dur = float(line.strip().split(" ")[2])
                        if dur < max_duration / 2:
                            train.append(line)

        for f in ['test']:
            for d in ['mdm', 'ihm', 'sdm']:
                with open(os.path.join(lists_path, f"ami-{d}-{f}.lst")) as lst:
                    for line in lst:
                        dur = float(line.strip().split(" ")[2])
                        if dur < max_duration / 2:
                            test.append(line)
        
        random.shuffle(train)
        random.shuffle(test)

        train_chunks = []
        test_chunks = []
        curr_chunk = []
        total_dur = 0
        while len(train) > 0:
            line = train.pop(0)
            dur = float(line.strip().split(" ")[2])
            if total_dur > max_duration:
                train_chunks.append(curr_chunk)
                curr_chunk = []
                total_dur = 0
            total_dur += dur
            curr_chunk.append(line)
        
        if len(curr_chunk) > 0:
            train_chunks.append(curr_chunk)
            curr_chunk = []
        total_dur = 0
        
        while len(test) > 0:
            line = test.pop(0)
            dur = float(line.strip().split(" ")[2])
            if total_dur > max_duration:
                test_chunks.append(curr_chunk)
                curr_chunk = []
                total_dur = 0
            total_dur += dur
            curr_chunk.append(line)

        if len(curr_chunk) > 0:
            test_chunks.append(curr_chunk)
            curr_chunk = []
            
        with Pool(processes) as p:           
            test_data = list(
                tqdm(
                    p.imap(to_list, test_chunks),
                    total=len(test_chunks),
                )
            )

        with Pool(processes) as p:           
            train_data = list(
                tqdm(
                    p.imap(to_list, train_chunks),
                    total=len(train_chunks),
                )
            )


        with open(train_file, "w") as lst, open(test_file, "w") as lst_test:
            lst.writelines(train_data)
            lst_test.writelines(test_data)
    
    print("Prepared Combined AMI", flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Combined Dataset creation.")
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

    args=parser.parse_args()

    audio_path=os.path.join(args.dst, "audio")
    text_path=os.path.join(args.dst, "text")
    lists_path=os.path.join(args.dst, "lists")
    os.makedirs(f"{audio_path}/ami_combined", exist_ok=True)
    os.makedirs(text_path, exist_ok=True)
    os.makedirs(lists_path, exist_ok=True)

    prepare_ami(audio_path, text_path, lists_path, args.process)

