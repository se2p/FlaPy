#!/usr/bin/env python3
# This project is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This project is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this project.  If not, see <https://www.gnu.org/licenses>.
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from json import JSONDecodeError
from typing import List, Optional, Dict, Any, Tuple

import requests
from github.MainClass import Github
from github.Rate import Rate
from github.RateLimit import RateLimit


@dataclass
class PyPIResult:
    author: str
    url: str
    classifiers: List[str]
    downloads: Dict[str, int]
    keywords: str
    license: str
    releases: List[Tuple[str, str]]


@dataclass
class GitHubResult:
    forks: int
    issues: int
    open_issues: int
    open_pull_requests: int
    pull_requests: int
    stargazers: int
    subscribers: int
    watchers: int


def get_result_from_pypi(pypi_package: str) -> Optional[PyPIResult]:
    def get_json() -> Dict[str, Any]:
        try:
            response = requests.get(
                url=f"https://pypi.python.org/pypi/{pypi_package}/json", stream=True,
            )
        except requests.exceptions.ConnectionError as err:
            raise ConnectionError(err)
        data = response.json()
        return data

    def parse_fields_from_json(data: Dict[str, Any]) -> PyPIResult:
        info_data = data["info"]
        release_data = data["releases"]
        return PyPIResult(
            author=info_data["author"],
            classifiers=info_data["classifiers"],
            downloads=info_data["downloads"],
            keywords=info_data["keywords"],
            license=info_data["license"],
            releases=[
                (key, values[0]["upload_time"]) if len(values) > 0 else (key, -1)
                for key, values in release_data.items()
            ],
            url=f"https://pypi.org/project/{pypi_package}/",
        )

    try:
        json_data = get_json()
        return parse_fields_from_json(json_data)
    except JSONDecodeError:
        return None


def get_result_from_github(token: str, repo_id: str) -> GitHubResult:
    def wait_if_necessary() -> None:
        def wait(seconds: int) -> None:
            print(f"Wait for {seconds} for GitHub API", file=sys.stderr)
            time.sleep(seconds)
            print("Continue", file=sys.stderr)

        rate_limit: RateLimit = gh.get_rate_limit()
        rate: Rate = rate_limit.core
        if rate.remaining <= 10:
            reset_time: datetime = rate.reset
            wait_time = reset_time - datetime.utcnow()
            wait(int(wait_time.total_seconds()) + 10)

    gh = Github(login_or_token=token, user_agent="se2p/python")
    wait_if_necessary()
    repo = gh.get_repo(repo_id)
    forks = repo.forks_count
    stargazers = repo.stargazers_count
    subscribers = repo.subscribers_count
    watchers = repo.watchers_count
    open_issues = repo.open_issues_count
    issues = repo.get_issues(state="all").totalCount
    open_pull_requests = repo.get_pulls(state="open").totalCount
    pull_requests = repo.get_pulls(state="all").totalCount
    return GitHubResult(
        forks=forks,
        issues=issues,
        open_issues=open_issues,
        open_pull_requests=open_pull_requests,
        pull_requests=pull_requests,
        stargazers=stargazers,
        subscribers=subscribers,
        watchers=watchers,
    )


def main() -> None:
    with open(os.path.join("..", "repos_with_commit.csv")) as in_file:
        with open(os.path.join("..", "results.csv"), mode="w") as out_file:
            in_csv = csv.DictReader(in_file)
            field_names = [
                "project",
                "pypi_url",
                "pypi_author",
                "pypi_classifiers",
                "pypi_downloads",
                "pypi_keywords",
                "pypi_license",
                "pypi_releases",
                "gh_forks",
                "gh_issues",
                "gh_open_issues",
                "gh_open_pull_requests",
                "gh_pull_requests",
                "gh_stargazers",
                "gh_subscribers",
                "gh_watchers",
            ]
            out_csv = csv.DictWriter(out_file, fieldnames=field_names)
            out_csv.writeheader()
            for row in in_csv:
                print(f"Collecting for {row['Project']}...")
                user_name = row["Project"].split("/")[-2]
                repo_name = row["Project"].split("/")[-1]
                try:
                    pypi_result = get_result_from_pypi(repo_name)
                    github_result = get_result_from_github(
                        "12804bff38f69935eff34181a356df2c054f5f21",
                        f"{user_name}/{repo_name}",
                    )
                except BaseException:
                    print(f"    Failed for {row['Project']}")
                    continue
                if pypi_result is None or github_result is None:
                    print(f"    Skipping {row['Project']}")
                    continue
                out_csv.writerow(
                    {
                        "project": row["Project"],
                        "pypi_author": pypi_result.author,
                        "pypi_url": pypi_result.url,
                        "pypi_classifiers": pypi_result.classifiers,
                        "pypi_downloads": pypi_result.downloads,
                        "pypi_keywords": pypi_result.keywords,
                        "pypi_license": pypi_result.license,
                        "pypi_releases": pypi_result.releases,
                        "gh_forks": github_result.forks,
                        "gh_issues": github_result.issues,
                        "gh_open_issues": github_result.open_issues,
                        "gh_open_pull_requests": github_result.open_pull_requests,
                        "gh_pull_requests": github_result.pull_requests,
                        "gh_stargazers": github_result.stargazers,
                        "gh_subscribers": github_result.subscribers,
                        "gh_watchers": github_result.watchers,
                    }
                )


if __name__ == "__main__":
    main()
