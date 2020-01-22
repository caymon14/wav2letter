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
        segment = segment[start:end]
        
        os.makedirs(export_path, exist_ok=True)
        segment.export(filename, format="flac")
    else:
        print("flac file exists, skipping")
    return f"{name} {filename} {end-start}.0 {text}"
