# DuoRAT re-implementation experiment

This repository is a reimplementation attempt of the [DuoRAT paper](https://arxiv.org/abs/2010.11119) which targets the NLP task of translating **natural language** query to **SQL statement** by utilizing relation-aware transformers.

This project is assigned in scope of the *CS5134 - Natural Language Processing - Spring 2024* course under the instruction of [Dr. Tianyu Jiang](https://jiangtianyu.com/).

Our team consists of:
- Bao Huynh
- Triet Pham
- Simon Feist

# Setting up and installing dependencies

1. Ensure you have Python 3.9+ on your system

2. Create and activate a Python virtual environment

```bash
python -m venv .venv
. .venv/bin/activate
# Or . .venv/Scripts/activate if Windows
```

3. Installing dependencies

* If you have Nvida GPU compatible with [CUDA](https://en.wikipedia.org/wiki/CUDA)
```bash
pip install -r requirements-cuda.txt
```

* Else, if you use MacOS or don't have an Nvidia GPU
```bash
pip install -r requirement.txt
```

NOTE: If the provided `requirements-....txt` file somehow did not work for you and produce errors during installation, you can try manually installing the dependencies:

- First, clear your virtual environment by `python uninstall -r requirements-uninstall.txt`

- Then, follow instruction on https://pytorch.org to install the correct `pytorch` version

- Then, install additional dependencies
```bash
pip install spacy gdown asdl pyasdl ipykernel autopep8
python -m spacy download en_core_web_sm
```

4. Download and extract the [Spider dataset](https://yale-lily.github.io/spider) to the same folder as the codebase

You can also run our provided utility script, which will automatically download into `spider.zip` (using `gdown`) and extracts into a `spider` folder

```bash
python download_spider.py
```

# Project structure

- `preprocess.py`: Functions for reading, storing, and transforming the dataset
- `transformer.py`: Vanilla Transformer model that acts as a baseline by attempting to treat Text-to-SQL as a traditional language translatation task
- `transformer-tree-decoder.py`: Transformer model that use the [TranX](https://arxiv.org/abs/1810.02720) tree-based decoder framework (WIP)
- `grammars` folder: Relevant code to parse and unparse SQL statements using the SQL ASDL (Abstract Syntax Description Language)