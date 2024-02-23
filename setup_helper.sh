#!/bin/bash

mkdir lr_ws
cd lr_ws
mkdir virtualenv
mkdir src
cd virtualenv
echo "Creating venv..."
python3 -m venv lr
cd ..
. virtualenv/lr/bin/activate
echo "Upgrading pip..."
pip install --upgrade pip
cd src

echo "Cloning..."
git clone https://gitlab.com/crzz/lr_gym.git

echo "Installing dependencies"
cd ..
pip install -r src/lr_gym/requirements.txt
