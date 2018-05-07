#!/bin/bash
# installing multiplot2d

# add this directory to PYTHONPATH in ~/.bashrc
echo export PYTHONPATH=${PYTHONPATH}:"$(cd "$(dirname "$1")"; pwd)/$(basename "$1")" >> ~/.bashrc
source ~/.bashrc

# update matplotlib
pip install --upgrade matplotlib
rm -r ~/.cache/matplotlib

# get user directory for matplotlib config
user_dir=`python <<END
import matplotlib
user_dir = matplotlib.get_configdir()
print user_dir
END`

# create stylelib dir
mkdir $user_dir/stylelib/

# copy multiplot2d stylesheet there
cp examples/*.mplstyle $user_dir/stylelib/

# confirm
echo "multiplot2d configured!"

