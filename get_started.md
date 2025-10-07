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

## Dataset setup