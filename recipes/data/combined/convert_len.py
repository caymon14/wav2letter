from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined Dataset creation.")
    parser.add_argument(
        "--file",
        help="destination directory where to store data",
        default="./data_dir",
    )

    args=parser.parse_args()

    with open(args.file, "r") as list_f, open(f"{args.file}_new" , "w") as wr:
        for line in list_f:
            params = line.strip().split(" ")
            params[2] = str(float(params[2])*1000)
            text = " ".join(params)
            wr.write(f"{text}\n")
        
