#!/bin/bash

# Get folder where the install script is located
dname=$(realpath $(dirname $0))
cd $HOME
rm -rf .hdf5plot
mkdir .hdf5plot
cd .hdf5plot
python3 -m venv hdf5plotvenv
. hdf5plotvenv/bin/activate
if [[ "$(which python3)" != "$HOME/.hdf5plot/hdf5plotvenv/bin/python3" ]]; then
    echo "Failed to enter venv."
    exit 1
fi

echo "Created venv at $(which python3)" 
pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade wheel

pip install -r "${dname}/requirements.txt"
mkdir -p $HOME/.local/bin
cp "${dname}/hdf5plot" $HOME/.local/bin
cp "${dname}/hdf5_plotter.py" $HOME/.hdf5plot/
mkdir -p $HOME/.local/share/applications
cp "${dname}/crizz-hdf5plotter.desktop" $HOME/.local/share/applications