from typing import List, Dict, Tuple, Set, Any
import logging
import random
import re
import subprocess
import time

from flapy.utils import try_default
from tqdm import tqdm
import fire
import requests
from bs4 import BeautifulSoup
import pandas as pd


GITHUB_URL_REGEX = r'https?://(www\.)?github\.com/[^/]+/[^/]+/?$'


session = requests.Session()
session.mount("", requests.adapters.HTTPAdapter(max_retries=10))


def fetch_all_pypi_projects() -> List[str]:
    logging.info("fetching all pypi projects")
    r = requests.get("https://pypi.org/simple/").text

    logging.info("parsing pypi projects")
    proj_names = [
        a.text
        for a in BeautifulSoup(r, "html.parser").find_all("a")
    ]
    # Remove first and last element since they are empty strings (bf4 parsing)
    return proj_names


def _get_pypi_metadata(project_name: str) -> Tuple[str, str, Any]:
    """
    :project_name: PyPI project name
    :returns: (status, http_response_code, json)
    """
    try:
        response = requests.get(
            url=f"https://pypi.python.org/pypi/{project_name}/json", stream=True,
        )
        http_response_code = response.status_code
    except Exception as ex:
        return "fetching failed", "", type(ex)
    try:
        data = response.json()
    except Exception as ex:
        return "parsing failed", http_response_code, type(ex)
    return "successful", http_response_code, data


def _get_latest_pypi_tag(pypi_json: Dict) -> str:
    """Given a PyPI metadata json derived from `_get_pypi_metadata`, return the latest release
    """
    try:
        releases = pypi_json["releases"]
    except Exception:
        raise ValueError("data does not contain releases")

    releases = [
        (
            r_k,
            try_default(lambda: r_v[0]["upload_time"], Exception, None)
        )
        for r_k, r_v in releases.items()
    ]
    releases = [
        (release_tag, release_date)
        for release_tag, release_date in releases
        if release_date is not None
    ]
    if len(releases) == 0:
        raise ValueError("release list is empty")
    return max(releases, key=lambda x: x[1])[0]


def _is_valid_github_url(github_url: str) -> bool:
    return re.match(GITHUB_URL_REGEX, github_url) is not None


def _get_git_tags(url: str) -> List[str]:
    """
    :param url: GitHub URL
    :return: Set of all GitTag versions
    """

    # add pseudo credentials to avoid prompt
    search_url: str = url[:8] + "pseudocredentials:pseudocredentials@" + url[8:]

    # obtain tags, if not found return empty set
    try:
        data: str = str(
            subprocess.check_output(
                ["git", "ls-remote", "--tags", search_url], stderr=subprocess.DEVNULL
            )
        )
    except subprocess.CalledProcessError:
        return set()

    # parse git tags
    new_data: List[str] = data.split("\\n")
    tags: Set[str] = set()
    for d in new_data:
        git_tag = re.search("tags/.*[0-9]+", d)
        if git_tag is not None:
            git_tag: str = str(git_tag.group(0))
            git_tag = git_tag.split("/")[1]
            tags.add(git_tag)
    return sorted(tags)


def _match_pypi_git_tag(pypi_version, git_tags) -> str:
    for git_tag in git_tags:
        regex = "v*" + pypi_version
        if re.match(regex, git_tag):
            return git_tag


def resolve_url(url: str, max_retries=6, sleep_time=11) -> Tuple[str, int]:
    num_retries = 0
    try:
        response = session.get(url)
        while response.status_code == 429 and num_retries < max_retries:
            num_retries += 1
            logging.info(f"Sleeping {sleep_time}s due to 429 ({num_retries}/{max_retries})")
            time.sleep(sleep_time)
            response = session.get(url)
        if num_retries == max_retries:
            logging.error(f"sleeping didn't help for {url}")
        return response.url, response.status_code
    except Exception:
        return None, None


def mine(
    *,
    sample_size: int = None,
    random_seed: int = None,
    project_list_file=None,
    redirect_github_urls=True,
    remove_duplicates=True,
    remove_no_github_url_found=False
) -> pd.DataFrame:
    """

    :sample_size: randomly sample `sample_size` number of projects
    :returns: TODO

    """

    if random_seed is not None:
        random.seed(random_seed)

    # 1. Fetch all PyPI projects
    if project_list_file is None:
        projects = fetch_all_pypi_projects()
    else:
        with open(project_list_file) as f:
            projects = f.read().split("\n")

    # 2. Random sampling
    if sample_size is not None:
        projects = random.sample(projects, sample_size)

    # 3. Fetch data for each project
    logging.info("retrieving detailed project information")
    project_details = []
    for proj_name in tqdm(projects):

        # 4. Fetch PyPI details
        fetch_status, http_response_code, pypi_data = _get_pypi_metadata(proj_name)
        pypi_classifiers = try_default(lambda: pypi_data["info"]["classifiers"])
        latest_pypi_tag = try_default(lambda: _get_latest_pypi_tag(pypi_data))
        pypi_project_urls = try_default(lambda: pypi_data["info"]["project_urls"])

        # 5. Search for Github URL (+ to-lower + redirect)
        github_url = try_default(
            lambda: [url for _, url in pypi_project_urls.items() if _is_valid_github_url(url)][0]
        )
        github_url = try_default(lambda: github_url.lower())
        if redirect_github_urls:
            github_url, github_url_status = try_default(lambda: resolve_url(github_url))
        else:
            github_url_status = None
        github_url = try_default(lambda: github_url.lower())

        # 6. Fetch git tags
        git_tags: str = try_default(lambda: _get_git_tags(github_url))

        # 7. match PyPI and git tag
        matching_github_tag = try_default(
            lambda: _match_pypi_git_tag(pypi_version=latest_pypi_tag, git_tags=git_tags)
        )

        project_details.append({
            # FlaPy Input columns
            "Project_Name": proj_name,
            "Github_URL": github_url,
            "matching_github_tag": matching_github_tag,
            "PYPI_latest_tag": latest_pypi_tag,
            "funcs_to_trace": "",
            "tests_to_run": "",

            # other columns (ignored by FlaPy)
            "pypi_fetch_status": fetch_status,
            "pypi_http_response_code": http_response_code,
            "PYPI_classifiers": pypi_classifiers,
            "PYPI_project_urls": pypi_project_urls,
            "github_url_status": github_url_status,
            "git_tags": git_tags,
        })
    project_details = pd.DataFrame(project_details)

    # 8. Drop duplicates (some PyPI projects point to the same GitHub URL)
    if remove_duplicates:
        num_duplicates = len(project_details[
            project_details.duplicated("Github_URL") & (~project_details["Github_URL"].isna())
        ])
        logging.info(f"Dropped {num_duplicates} duplicated entries (same GitHub URL)")
        project_details = project_details[
            (~project_details.duplicated("Github_URL")) | (project_details["Github_URL"].isna())
        ]

    # 9. Remove cases where no Github URL was found
    if remove_no_github_url_found:
        project_details = project_details[~project_details["Github_URL"].isna()]

    return project_details


def main() -> None:
    fire.Fire()


if __name__ == "__main__":
    main()
