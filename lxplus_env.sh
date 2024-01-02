#!/bin/bash

TRAINPATH=`pwd`
export DEEPJETCORE=${TRAINPATH}/../DeepJetCore

if [ -z "$CMSSW_BASE" ]; then
    echo "Setting up the conda-based environment"
    export PATH=/afs/cern.ch/user/p/psilva/work/Wmass/train/miniconda/bin:$PATH
    cd $DEEPJETCORE/
    source lxplus_env.sh
    cd $TRAINPATH
fi


export PYTHONPATH="${TRAINPATH}/Train:${TRAINPATH}/modules:${DEEPJETCORE}/../:${PYTHONPATH}"
export LD_LIBRARY_PATH=${DEEPJETCORE}/compiled:$LD_LIBRARY_PATH