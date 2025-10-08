# Project setup

0. `git clone https://github.com/TheLion-ai/UMIE_datasets`
1. Install `poetry`

    - [`poetry` official source](https://python-poetry.org/docs/#installing-with-the-official-installer)

2. Add `poetry` to PATH
    - default path on Unix `$HOME/local/bin`, add it to `.bashrc`
    - default path on windows `%APPDATA%\Python\Scripts` on Windows
3. run `poetry install`
4. create python virtual environment: 
    - with `poetry env use python3.12` or `python -m venv .local-venv`
5. activate the virtual environment
    - `poetry env activate` or `source ./.local-venv/bin/activate`

------------------------

## Dataset preparation

0. create new directory for project datasets: `mkdir datasets`
1. to clone submodules run: `git submodule update --init --recursive`
    - This command initializes any uninitialized submodules, updates existing ones to the correct commit, and fetches new commits from their remotes. The `--init` flag ensures new submodules are registered and cloned, while `--recursive` handles nested submodules. 
  
2. build `kits23`
    - Enter the KITS-23 (datasets/kite23) directory and install the packages with pip.
        ```bash
        cd datasets/kits23
        pip3 install -e .

        kits23_download_data # download data in `datasets/kit23/dataset/` directory
        ```



3. download from Kaggle run: `python download_kaggle.py`
    - console will output the path where the datasets are stored. They are need for `config/runner_config.py`