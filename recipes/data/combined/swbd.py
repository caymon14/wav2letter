from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
from multiprocessing import Pool
import re

import numpy
from tqdm import tqdm
from utils import remove_punct, convert_to_flac
from functools import partial
from glob import glob
from pydub import AudioSegment
import subprocess
from collections import defaultdict

alpha = re.compile(r"^[a-zA-Z\s]+$") 
max_duration = 10 * 1000
test_every = 5

def cure_text(text):
    text = remove_punct(text)
    text = text.lower()
    return " ".join(map(lambda x: x.strip(), text.split(" ")))

def read_dialogs(name, transcript_map):
    dialogs = []
    for line in transcript_map[name]:
        file_id, text = line.split()
        metadata = file_id.split("-")[1]
        speaker = metadata.split("_")[0]

        start = float(metadata.split("_")[1])
        end = float(metadata.split("-")[1])
        channel = 1
        if speaker == "B":
            channel = 2

        dialogs.append((text, start, end, channel))

    return dialogs

def swbd_to_list(audio_path, swbd_path, sph2pipe, transcript_map, sph_file):
    name = os.path.basename(sph_file).split(".")[0]
    dialogs = read_dialogs(name, transcript_map)

    export_dir = f"{audio_path}/swbd/"
    
    for channel in ["1", "2"]:
        wav_file = f"{swbd_path}/{name}_c{channel}.wav"
        subprocess.check_call([sph2pipe, "-c", channel, "-p", "-f", "rif", sph_file, wav_file])

    lists = []
    for i, (text, start, end, channel) in enumerate(dialogs):
        if alpha.match(text):
            lst_record = convert_to_flac(f"{swbd_path}/{name}_c{channel}.wav",
                                        start, end, f"{name}_{i}", export_dir, text)
            lists.append(lst_record)
    for channel in ["1", "2"]:   
        os.remove(f"{swbd_path}/audio/{scenario}/{name}_c{channel}.wav")
    return lists
            

def prepare_swbd(swbd, audio_path, text_path, lists_path, processes, sph2pipe):
    train_file = f"{lists_path}/swbd-train.lst"
    transcript_map = defaultdict()
    with open(f"{swbd}/text", "r") as train_f:
        for line in train_f.readlines():
            name = line.split()[0].split("-")[0]
            transcript_map[name].append(line)
    if not os.path.exists(train_file):
        with Pool(processes) as p:
            files = list(glob(f"{swbd}/**/*.sph"))
            to_list = partial(swbd_to_list, audio_path, swbd, sph2pipe, transcript_map)
            samples = list(
                tqdm(
                    p.imap(to_list, files),
                    total=len(files),
                )
            )
        with open(train_file, "w") as list_f:
            for s in samples:
                list_f.writelines(s)

    else:
        print(f"{train_file} exists, doing verify")
        new_list = []
        with open(train_file, "r") as list_f:
            for line in list_f:
                filename = line.split(" ")[1]
                text = " ".join(line.strip().split(" ")[3:])
                params = " ".join(line.strip().split(" ")[:3])
                line = f"{params} {text}\n"
                if not os.path.exists(filename) or len(text) < 2 or not alpha.match(text):
                    print(f"{filename} does not exists or text is empty, text: {text}")
                else:
                    new_list.append(line)
        with open(train_file, "w") as list_f:
            list_f.writelines(new_list)
    
    print("Prepared swbd", flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Combined Dataset creation.")
    parser.add_argument(
        "--dst",
        help="destination directory where to store data",
        default="./data_dir",
    )
    parser.add_argument(
        "--swbd",
        help="swbd data location",
        default="./swbd",
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
    os.makedirs(f"{audio_path}/swbd", exist_ok=True)
    os.makedirs(text_path, exist_ok=True)
    os.makedirs(lists_path, exist_ok=True)

    prepare_swbd(args.swbd, audio_path, text_path, lists_path, args.process, args.sph2pipe)

