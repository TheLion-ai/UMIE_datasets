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
    - define `KITS23` variable with the full path to `datasets/kits23` in `.pipeline.env` file:
            ex: `KITS23=/home/user/UMIE_datasets/datasets/kits23/dataset`



3. download from Kaggle run: `python download_kaggle.py`
    - this script will download all kaggle's repo and generate a `.pipeline.env`


----
    Manual downloads
----
1. download [Chest X-ray 14](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345)
    - create an account
    - download the `images` folder and `DataEntry2017_v2020.csv` (warning `images` is 45.1GB!)
    - create a new folder inside `datasets` run: `mkdir datasets/chestxray14`
    - move the downloaded extracted `images` and `DataEntry2017_v2020.csv` into `datasets/chestxray14`
    - define `CHESTXRAY14` variable with the full path to `datasets/chestxray14` in `.pipeline.env` file:
        ex: `CHESTXRAY14=/home/user/UMIE_datasets/datasets/chestxray14`

2. NOTE: `azcopy` REQUIRE [work or school account](https://learn.microsoft.com/en-us/answers/questions/5320873/you-cant-sign-in-here-with-a-personal-account-use)
    - install [`azcopy`](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10).
    - on linux
        ```bash
            wget https://aka.ms/downloadazcopy-v10-linux -O azcopy_v10.tar.gz
            tar -xvf azcopy_v10.tar.gz
            sudo mv azcopy_linux_amd64_10.30.1/azcopy /usr/bin/
            azcopy --version
        ```
    - or you can download the portable binary on microsoft's website
    - to download the data collection run: `azcopy copy "./datasets/coca" "[YOUR_URL from Standford's website]"`
    - Go to [COCA- Coronary Calcium and chest CTs](https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa).
        1. Log in or sign up for a Stanford AIMI account.
        2. Fill in your contact details.
    Go to [BrainMetShare](https://aimi.stanford.edu/brainmetshare).
        1. Log in or sign up for a Stanford AIMI account.
        2. Fill in your contact details.

3. CT-ORG (unable to verify because IBM Aspera Client is broken)
  1. Go to [CT-ORG](https://www.cancerimagingarchive.net/collection/ct-org/) page on Cancer imaging archive.
  2. Download the data.
  3. Extract `PKG - CT-ORG`.

4. Install [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)
    A. **LIDC-IDRI**
        - run/click: `nbia_manifests/TCIA_LIDC-IDRI_20200921.tcia` to download ['Images'](https://www.cancerimagingarchive.net/collection/lidc-idri/)
        - Download ['Radiologist Annotations/Segmentations'](https://www.cancerimagingarchive.net/collection/lidc-idri/)
        - extract to `datasets/LIDC`
        - define `LIDC_MANIFEST` variable with the full path to `datasets/LIDC` in `.pipeline.env` file:
            ex: `LIDC_MANIFEST=/home/user/UMIE_datasets/datasets/LIDC/manifest-1600709154662/LIDC-IDRI`
        - define `LIDC_XML` variable with the full path to `datasets/LIDC` in `.pipeline.env` file:
            ex: `LIDC_XML=/home/user/UMIE_datasets/datasets/LIDC/LIDC-XML-Only`

    B. **Chinese Mammography Database**
        - run/click: `nbia_manifests/The-Chinese-Mammography-Database.tcia` to download ['Images'](https://www.cancerimagingarchive.net/collection/cmmd/)
        - Download ['Clinical Data'](https://www.cancerimagingarchive.net/collection/cmmd/)
        - extract both to `datasets/cmmd`
          - define `CMMD_MANIFEST` variable with the full path to `datasets/cmmd` in `.pipeline.env` file:
            ex: `CMMD_MANIFEST=/home/user/UMIE_datasets/datasets/cmmd/manifest-1616439774456/CMMD`
        - define `CMMD_CLINICAL` variable with the full path to `datasets/cmmd` in `.pipeline.env` file:
            ex: `CMMD_CLINICAL=/home/user/UMIE_datasets/datasets/cmmd/CMMD_clinicaldata_revision.xlsx`



----------------

#### See `pipeline.env.example` 