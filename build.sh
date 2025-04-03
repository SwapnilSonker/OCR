#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt
python -m spacy download en_core_web_trf