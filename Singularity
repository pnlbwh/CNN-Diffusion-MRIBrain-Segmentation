Bootstrap: docker
From: redhat/ubi9:9.5-1738643550

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
    yum -y install wget file bzip2 which vim git make libstdc++-static unzip mesa-libGL bc libSM \
    gcc-c++ openssl-devel libX11-devel && \
    yum clean all && \
    #
    # install miniconda3
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p miniconda3/ && \
    source miniconda3/bin/activate && \
    #
    # create conda environment
    git clone https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation.git && \
    cd CNN-Diffusion-MRIBrain-Segmentation && \
    wget https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation/releases/download/v0.3/model_folder.tar.gz && \
    tar -xzvf model_folder.tar.gz && rm -f model_folder.tar.gz && \
    conda create -y -n dmri_seg python=3.11 -c conda-forge --override-channels && \
    conda activate dmri_seg && \
    pip install 'tensorflow[and-cuda]==2.15.1' && \
    pip install scikit-image git+https://github.com/pnlbwh/conversion.git && \
    cd
    #
    # install FSL
    echo "Downloading FSL installer" && \
    wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py -O fslinstaller.py > /dev/null 2>&1 && \
    echo "Installing FSL" && \
    V=6.0.7 && \
    python fslinstaller.py -V $V -d $HOME/fsl-$V > /dev/null && \
    rm -f fslinstaller.py && \
    #
    # install CMake
    CMAKE=3.31.0 && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE}/cmake-${CMAKE}.tar.gz && \
    tar -xzf cmake-${CMAKE}.tar.gz && \
    cd cmake-${CMAKE} && mkdir build && cd build && \
    ../bootstrap --parallel=4 && make -j4
    cd
    #
    # install dcm2niix
    git clone https://github.com/rordenlab/dcm2niix.git && \
    cd dcm2niix && mkdir build && cd build && \
    /home/pnlbwh/cmake-${CMAKE}/build/bin/cmake .. && make -j4 && \
    mv bin/dcm2niix /usr/bin/ && \
    cd && rm -rf $HOME/dcm2niix && \
    #
    # install ANTs
    git clone https://github.com/ANTsX/ANTs.git
    cd ANTs && mkdir build && cd build
    /home/pnlbwh/cmake-${CMAKE}/build/bin/cmake .. && make -j4
    cd
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
    #
    # setup FSL
    export FSLDIR=/home/pnlbwh/fsl-6.0.7
    export FSLOUTPUTTYPE=NIFTI_GZ
    export PATH=$FSLDIR/share/fsl/bin:$PATH
    #
    # add dcm2niix and dwi_masking.py to PATH
    export PATH=/home/pnlbwh/dcmbin:/home/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation/pipeline:$PATH
    #
    # setup ANTSPATH
    export ANTSPATH=/home/pnlbwh/ANTs/build/ANTS-build/Examples
    export PATH=$ANTSPATH:/home/pnlbwh/ANTs/Scripts:$PATH
    #
    # dmri_seg conda environment
    export PATH=/home/pnlbwh/miniconda3/envs/dmri_seg/bin:$PATH


