#!/usr/bin/env python3
from typing import List, Tuple

import pandas as pd

# from pandarallel import pandarallel
import requests
from requests.adapters import HTTPAdapter
import sys

# pandarallel.initialize(nb_workers=8)

session = requests.Session()
session.mount("", HTTPAdapter(max_retries=10))


def resolve_url(url: str) -> Tuple[str, int]:
    response = session.get(url)
    print(".", end="", flush=True, file=sys.stderr)
    return response.url, response.status_code


def main(args: List[str] = None):
    if args is None:
        args = sys.argv[1:]

    if len(args) != 1:
        print("USAGE: remove_duplicate_urls.py REPOS_CSV", file=sys.stderr)
        print(file=sys.stderr)
        print(
            "REPOS_CSV should have two columns (project_name, url) but no header row!",
            file=sys.stderr,
        )
        sys.exit()

    df = pd.read_csv(args[0], names=["project", "url"])
    print(f"Input csv has {len(df)} rows", file=sys.stderr)

    print(
        f"Removed {len(df[df.duplicated(subset='url')])} rows that were already duplicated in the input data",
        file=sys.stderr,
    )
    df.drop_duplicates(subset="url", inplace=True)

    print("Resolving URLs", end="", file=sys.stderr)
    df["url"], df["status"] = list(zip(*df["url"].apply(resolve_url)))
    print(file=sys.stderr)

    # print(file=sys.stderr)
    # print(
    #     f"Removed {len(df[df['status'] != 200])} rows that had response-status != 200:",
    #     file=sys.stderr,
    # )
    # print(df[df['status'] != 200], file=sys.stderr)
    # df = df[df['status'] == 200]
    df = df[["project", "url"]]

    print(file=sys.stderr)
    print(
        f"Removed {len(df[df.duplicated()])} rows that redirected to the same url:",
        file=sys.stderr,
    )
    print(df[df.duplicated()], file=sys.stderr)
    df.drop_duplicates(subset="url", inplace=True)

    session.close()

    df.to_csv(sys.stdout, index=False, header=False)


if __name__ == "__main__":
    main()
