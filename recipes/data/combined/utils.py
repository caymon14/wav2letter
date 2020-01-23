"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os

import sox
from pydub import AudioSegment
import csv


def find_transcript_files(dir):
    files = []
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(".trans.txt"):
                files.append(os.path.join(dirpath, filename))
    return files


def parse_speakers_gender(spk_file):
    ret = {}
    with open(spk_file, "r") as f:
        for line in f:
            if line.startswith(";"):
                continue
            sample_id, gen, _ = line.split("|", 2)
            ret[sample_id.strip()] = gen.strip()
    return ret


def transcript_to_list(file):
    audio_path = os.path.dirname(file)
    ret = []
    with open(file, "r") as f:
        for line in f:
            file_id, trans = line.strip().split(" ", 1)
            audio_file = os.path.abspath(
                os.path.join(audio_path, file_id + ".flac"))
            duration = sox.file_info.duration(audio_file) * 1000  # miliseconds
            ret.append([file_id, audio_file, str(duration), trans.lower()])

    return ret


def read_list(src, files):
    ret = []
    for file in files:
        with open(os.path.join(src, file + ".lst"), "r") as f:
            for line in f:
                _, _, _, trans = line.strip().split(" ", 3)
                ret.append(trans)

    return ret


def read_txt(file):
    ret = []
    with open(file) as f:
        for line in f:
            name, text = line.strip().split(" ", 1)
            ret.append((name, text))
    return ret


def convert_to_flac(file, start, end, name, export_path, text):
    filename = f"{export_path}/{name}.flac"
    if not os.path.exists(filename):
        segment = AudioSegment.from_file(file)
        if start is not None and end is not None:
            segment = segment[start:end]
        else:
            start = 0
            end = segment.duration_seconds * 1000

        os.makedirs(export_path, exist_ok=True)
        segment = segment.set_sample_width(2)
        segment = segment.set_frame_rate(16000)
        segment.export(filename, format="flac")
    else:
        print("flac file exists, skipping")
    return f"{name} {filename} {end-start}.0 {text}\n"


def read_tsv(file):
    rows = []
    with open(file) as f:
        reader = csv.DictReader(f, dialect="excel-tab")
        for row in reader:
            rows.append(row)

    return rows


def commonvoice_to_list(audio_path, f, commonvoice_location, line):
    path = line['path']
    text = line['sentence']
    text = text.lower()
    text = text.replace("\"", "")
    text = text.replace("!", "")
    text = text.replace("?", "")
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("’", "")
    text = text.replace("‘", "")

    export_dir = f"{audio_path}/commonvoice/{f}"
    lst_record = convert_to_flac(f"{commonvoice_location}/clips/{path}",
                                 None, None, path, export_dir, text)
    return lst_record


def ami_ihm_to_list(audio_path, ami_ihm_location, line):
    name, text = line
    _, scenario, headphone, _, start, end = name.split("_")
    export_dir = f"{audio_path}/ihm/{scenario}"
    lst_record = ""
    try:
        lst_record = convert_to_flac(f"{ami_ihm_location}/{scenario}/audio/{scenario}.Headset-{int(headphone[2:])}.wav",
                                    int(start)*10, int(end)*10, name, export_dir, text)
    except:
        print(f"Cannot convert file {name}")
    return lst_record


def ami_sdm_to_list(audio_path, ami_sdm_location, line):
    name, text = line
    _, scenario, _, _, start, end = name.split("_")
    export_dir = f"{audio_path}/sdm/{scenario}"
    lst_record = ""
    try:
        lst_record = convert_to_flac(f"{ami_sdm_location}/{scenario}/audio/{scenario}.Array1-01.wav",
                                    int(start)*10, int(end)*10, name, export_dir, text)
    except:
        print(f"Cannot convert file {name}")
    return lst_record


def ami_mdm_to_list(audio_path, ami_mdm_location, line):
    name, text = line
    _, scenario, _, _, start, end = name.split("_")
    export_dir = f"{audio_path}/mdm/{scenario}"
    lst_record = convert_to_flac(f"{ami_mdm_location}/{scenario}/{scenario}_MDM8.wav",
                                 int(start)*10, int(end)*10, name, export_dir, text)
    return lst_record


def ted_to_list(audio_path, f, ted_location, line):
    name, text = line
    scenario, start, end = name.split("-")
    export_dir = f"{audio_path}/ted/{scenario}"
    lst_record = convert_to_flac(f"{ted_location}/legacy/{f}/sph/{scenario}.sph",
                                 int(start)*10, int(end)*10, name, export_dir, text)
    return lst_record

def remove_punct(text):
    text = text.replace("<unk>", "")
    text = text.replace("\"", "")
    text = text.replace("!", "")
    text = text.replace("?", "")
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("’", "")
    text = text.replace("‘", "")
    text = text.replace("<", "")
    text = text.replace(">", "")
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("-", " ")
    text = text.replace("_", " ")

    return text
