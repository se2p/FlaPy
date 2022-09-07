#!/usr/bin/env python3
from typing import List, Tuple
from time import sleep

import pandas as pd
from tqdm import tqdm

import requests
from requests.adapters import HTTPAdapter
import sys

session = requests.Session()
session.mount("", HTTPAdapter(max_retries=10))


def resolve_url(url: str) -> Tuple[str, int]:
    num_retries = 0
    num_max_retries = 5
    sleep_time = 11
    response = session.get(url)
    while response.status_code == 429 and num_retries < num_max_retries:
        num_retries += 1
        sleep(sleep_time)
        print(f"Sleeping due to 429 ({num_retries})", file=sys.stderr)
        response = session.get(url)
    if num_retries == num_max_retries:
        print("ERROR: sleeping didn't help", file=sys.stderr)
    return response.url, response.status_code


def main(args: List[str] = None):
    if args is None:
        args = sys.argv[1:]

    if len(args) != 2:
        print("USAGE: python redirect_over_csv.py REPOS_CSV OUTPUT_CSV", file=sys.stderr)
        print(file=sys.stderr)
        print(
            "REPOS_CSV should have two columns (Project_Name, Project_URL)",
            file=sys.stderr,
        )
        sys.exit()

    # -- PARSE ARGUMENTS
    input_csv, output_csv = args

    # -- READ CSV
    df = pd.read_csv(input_csv)
    print(f"Input csv has {len(df)} rows", file=sys.stderr)

    # -- Resolve URLs
    #  do not try to parallelize this, it will just give you a 429 (too many requests)
    print("Resolving URLs", end="", file=sys.stderr)
    df["response"] = [resolve_url(url) for url in tqdm(df["Project_URL"])]

    # -- Prettify Dataframe
    df["Project_URL_redirected"] = df["response"].apply(lambda x: x[0])
    df["Project_URL_status"] = df["response"].apply(lambda x: x[1])
    df["same_redirect"] = (df["Project_URL"] == df["Project_URL_redirected"])
    df = df.drop(columns="response")

    session.close()

    # -- Write output
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
