FROM registry.hub.docker.com/library/python:3.10.1-bullseye AS build

WORKDIR /flapy_build

ENV POETRY_VERSION="1.8.3"

# Python shall not write the byte code to *.pyc files; they cannot be cached between
# runs of the container anyway, hence we save the required time and resources for that
ENV PYTHONDONTWRITEBYTECODE 1
# Prevent Python from buffering output that is written to STDOUT/STDERR; this allows to
# monitor the output in real time
ENV PYTHONUNBUFFERED 1

COPY pyproject.toml poetry.lock README.md ./
COPY ./flapy ./flapy

RUN pip install "poetry==${POETRY_VERSION}" \
    && poetry config virtualenvs.create false \
    && poetry build


# -- EXECUTE --
FROM registry.hub.docker.com/library/python:3.10.1-bullseye AS execute

ENV FLAPY_VERSION="0.2.0"

WORKDIR /workdir

RUN apt-get update && apt-get install -y sqlite3 cloc

COPY --from=build /flapy_build/dist/flapy-${FLAPY_VERSION}-py3-none-any.whl ./
COPY utils.sh clone_and_run_tests.sh findpackages.py find_all_modules.py ./

RUN pip install flapy-${FLAPY_VERSION}-py3-none-any.whl

ENTRYPOINT ["./clone_and_run_tests.sh"]
