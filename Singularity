Bootstrap: docker
From: centos:7.5.1804

%labels
    MAINTAINER Tashrif Billah <tbillah@bwh.harvard.edu>

%help
    https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation 

    Please report issues on GitHub.


%post
    #
    # set up user and working directory
    mkdir /home/pnlbwh
    cd /home/pnlbwh
    export HOME=`pwd`
    #
    # install required libraries
    yum -y groupinstall 'development tools' && \
    yum -y install wget file bzip2 which vim git make libstdc++-static mesa-libGL bc tcsh libSM && \
    yum clean all && \
    #
    # install FSL
    echo "Downloading FSL installer" && \
    wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py -O fslinstaller.py > /dev/null 2>&1 && \
    echo "Installing FSL" && \
    python fslinstaller.py -V 6.0.1 -d $HOME/fsl-6.0.1 -p > /dev/null 2>&1 && \
    # setup FSL environment
    export FSLDIR=$HOME/fsl-6.0.1 && \
    source $FSLDIR/etc/fslconf/fsl.sh && \
    $FSLDIR/fslpython/bin/conda clean -y --all && \
    rm -f fslinstaller.py && \
    #
    # install miniconda3
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p miniconda3/ && \
    source miniconda3/bin/activate && \
    #
    # create conda environment
    git clone https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation.git && \
    conda env create -f CNN-Diffusion-MRIBrain-Segmentation/environment_cpu.yml && \
    #
    # install dcm2niix
    conda install -c anaconda cmake && \
    git clone https://github.com/rordenlab/dcm2niix.git && \
    cd dcm2niix && mkdir build && cd build && \
    cmake .. && make -j4 && \
    mv bin $HOME/dcmbin/ && \
    cd && rm -rf $HOME/dcm2niix && \
    #
    # clean up
    rm -rf $HOME/.cache/pip/ $HOME/Miniconda3-latest-Linux-x86_64.sh && \
    conda clean -y --all && \
    #
    # change permission so a user is able to run tests
    # to be able to run tests, increase tmpfs size in /etc/singularity/singularity.conf
    # sessiondir max size = 8000
    chmod -R o+w /home/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation/tests/
    
%environment
    #
    # set up bashrc i.e shell
    # dmri_seg conda environment
    export CONDA_SHLVL=2
    export CONDA_PROMPT_MODIFIER=(dmri_seg)
    export CONDA_EXE=/home/pnlbwh/miniconda3/bin/conda
    export _CE_CONDA=
    export CONDA_PREFIX_1=/home/pnlbwh/miniconda3
    export CONDA_PREFIX=/home/pnlbwh/miniconda3/envs/dmri_seg
    export CONDA_PYTHON_EXE=/home/pnlbwh/miniconda3/bin/python
    export CONDA_DEFAULT_ENV=dmri_seg
    export PATH=/home/pnlbwh/miniconda3/envs/dmri_seg/bin:/home/pnlbwh/miniconda3/condabin:$PATH
    #
    # setup FSL
    export FSLDIR=/home/pnlbwh/fsl-6.0.1
    source $FSLDIR/etc/fslconf/fsl.sh
    export PATH=$FSLDIR/bin:$PATH
    #
    # add dcm2niix and dwi_masking.py to PATH
    export PATH=/home/pnlbwh/dcmbin:/home/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation/pipeline:$PATH
    #
    # setup ANTSPATH
    export ANTSPATH=/home/pnlbwh/miniconda3/envs/dmri_seg/bin

