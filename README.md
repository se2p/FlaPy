# Flaky Analysis

Flaky Analysis is a small tool that allows software developers
to automatically check tests for flakiness.

It is the result of research carried out at the
[Chair of Software Engineering II](https://www.fim.uni-passau.de/lehrstuhl-fuer-software-engineering-ii/)
at the [University of Passau](https://www.uni-passau.de), Germany.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python in at least version 3.7.
- You have installed the latest version of [`poetry`](https://python-poetry.org).
    - `pip install poetry`
- You have a Linux machine with a recent kernel and activated cgroups.
  We refer you to the installation documentation of
  [`BenchExec`](https://github.com/sosy-lab/benchexec),
  a framework for reliable benchmarking and resource measurement,
  which we use for running all test cases in isolation.


## Installing from source

```bash
git clone https://github.com/se2p/FlaPy.git
cd FlaPy
poetry install
poetry build
```

## Using Flaky Analysis

```bash
             # Drop first line, which contains the header
./run_csv.sh <(tail -n+2 sample_input.csv)
```

Results can then be found under `./flapy-results/`

To parse the results, use

```bash
poetry run results_parser ResultsDir --dir=flapy-results get_passed_failed to_csv --index=False > passed_failed_sample.csv

poetry run results_parser PassedFailed --file_ passed_failed_sample.csv to_test_overview to_csv --index=False > test_overview_sample.csv
```

An overview of the results can now be found in `passed_failed_sample.csv` and `test_overview_sample.csv`


## Building the project

The project can be built by using the `poetry` tool:
```bash
poetry build
```
This command will build two files in the `dist` folder:
A `tar.gz` archive and a `whl` Python wheel file.

## Contributors

See the contributors list

## Contact

If you want to contact me,
please find my contact details on my
[page at the University of Passau](https://www.fim.uni-passau.de/lehrstuhl-fuer-software-engineering-ii/lehrstuhlteam/?target=117242&module=TemplatePersondetails&config_id=a8d16e612076595e5b55aa227262539d&range_id=l13&username=lukasc01&group_id=).

## License

This project is licensed under the terms of the GNU Lesser General Public License.
