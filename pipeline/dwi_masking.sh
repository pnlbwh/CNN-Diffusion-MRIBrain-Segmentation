#!/bin/bash

SCRIPTDIR=$(dirname $0)

export LD_LIBRARY_PATH=PREFIX/lib
PREFIX/bin/python $SCRIPTDIR/dwi_masking.py $@

