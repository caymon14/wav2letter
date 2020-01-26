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

CALLHOME_URL = "https://media.talkbank.org/ca/CallHome/eng"

alpha = re.compile(r"^[a-zA-Z\s]+$") 
max_duration = 10 * 1000
test_every = 5

def cure_text(text):
    text = re.sub(r"&=\S+", "", text)
    text = re.sub(r"\[.+?\]", "", text)
    text = re.sub(r"@s:\S+", "", text)

    text = text.replace("+&", "")
    text = text.replace("xxx", "")
    text = text.replace("0", "")
    text = text.replace("&", "")
    text = text.replace("☺", "")
    text = text.replace("▔", "")
    text = text.replace("\n", " ")
    
    text = remove_punct(text)
    text = text.lower()
    return " ".join(map(lambda x: x.strip(), text.split(" ")))

def read_dialogs(lines):
    dialogs = []
    accum_dialog = ""
    started = False
    for line in lines:
        if line.startswith("*"):
            started = True
        if line.startswith("%"):
            started = False
        if started:          
            _, text = line.split("\t")
            text_and_timings = text.split("\x15")
            if len(text_and_timings) == 3:
                text, timings, _ = text_and_timings
                text = f"{accum_dialog} {text.strip()}"
                start, end = timings.split("_")
                start = int(start)
                end = int(end)
                started = False
                dialogs.append((text, start, end))
                accum_dialog = ""
            else:
                accum_dialog = f"{accum_dialog} {text.strip()}".strip()

    return dialogs
            

def prepare_callhome(callhome, audio_path, text_path, lists_path, processes):
    train_file = f"{lists_path}/callhome-train.lst"
    test_file = f"{lists_path}/callhome-test.lst"
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        with open(train_file, "w") as lst, open(test_file, "w") as lst_test:
            for file in glob(f"{callhome}/*.cha"):
                mp3_name = os.path.basename(file).split(".")[0]
                original_audio = AudioSegment.from_file(f"{callhome}/{mp3_name}.mp3")
                original_audio = original_audio.set_sample_width(2)
                original_audio = original_audio.set_frame_rate(16000)
                original_audio = original_audio.set_channels(1)
                accum_duration = 0
                current_pointer = 0
                accum_text = []
                with open(file, "r") as f:
                    dialogs = read_dialogs(f.readlines())
                    writes = 0
                    for text, start, end in dialogs:
                        if current_pointer == 0:
                            current_pointer = start
                        
                        accum_duration = end - current_pointer
                        accum_text.append(text.strip())
                        if accum_duration >= max_duration:
                            filename = f"{mp3_name}_{end-accum_duration}_{end}.flac"
                            filepath = f"{audio_path}/callhome/{filename}"
                            final_text = cure_text(" ".join(accum_text)).strip()
                            if writes == 5:
                                lst_test.write(f"{filename} {filepath} {accum_duration}.0 {final_text}\n")
                            else:
                                lst.write(f"{filename} {filepath} {accum_duration}.0 {final_text}\n")
                            save_segment = original_audio[current_pointer:end]
                            save_segment.export(filepath, format="flac")
                            accum_text = []
                            current_pointer = end
                            writes += 1
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
    
    print("Prepared CallHome", flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Combined Dataset creation.")
    parser.add_argument(
        "--dst",
        help="destination directory where to store data",
        default="./data_dir",
    )
    parser.add_argument(
        "--callhome",
        help="Callhome data location",
        default="./CallHome",
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
    os.makedirs(f"{audio_path}/callhome", exist_ok=True)
    os.makedirs(text_path, exist_ok=True)
    os.makedirs(lists_path, exist_ok=True)

    prepare_callhome(args.callhome, audio_path, text_path, lists_path, args.process)

