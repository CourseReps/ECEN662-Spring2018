#!/bin/bash
# installing quatematics

# add this directory to PYTHONPATH in ~/.bashrc
echo export PYTHONPATH=${PYTHONPATH}:"$(cd "$(dirname "$1")"; pwd)/$(basename "$1")" >> ~/.bashrc
source ~/.bashrc
