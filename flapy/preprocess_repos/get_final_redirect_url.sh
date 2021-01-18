#!/usr/bin/env bash

curl -I -Ls -o /dev/null -w %{url_effective} $1

