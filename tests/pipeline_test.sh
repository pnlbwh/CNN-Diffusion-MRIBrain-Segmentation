#!/usr/bin/env bash

# ================================
SCRIPT=$(readlink -m $(type -p $0))
SCRIPTDIR=$(dirname $SCRIPT)
testDataDir=$SCRIPTDIR/data
prismaDir=$SCRIPTDIR/data/Prisma
testDir=$SCRIPTDIR/
zipFile=dtitest_Siemens_SC.zip

# ================================
echo Downloading Prisma data

if [ ! -f $testDataDir/$zipFile ]
then
    wget http://people.cas.sc.edu/rorden/SW/dcm2niix/$zipFile -P $testDataDir
else
    echo $testDataDir/$zipFile exists, unzipping $zipFile
fi

rm -rf $prismaDir
unzip -d $testDataDir $testDataDir/$zipFile 


pushd .
cd $prismaDir
dcm2niix -o . -f %z -z y .

IMAGELIST=$testDataDir/imagelist.txt
MASKLIST=$testDataDir/masklist.txt
B0LIST=$testDataDir/b0list.txt
WarpedB0LIST=$testDataDir/warped_b0list.txt
WarpedMASKLIST=$testDataDir/warped_masklist.txt
CASELIST=$testDataDir/caselist.txt

if [ -f $IMAGELIST ]
then
    rm $IMAGELIST $MASKLIST $B0LIST $WarpedB0LIST $WarpedMASKLIST $CASELIST
fi


echo 'Generating b0 image and corresponding brain mask for test data ...'
for i in `ls *.nii.gz`
do
    IFS=., read -r prefix _ _ <<< $i
    if [ -f $prefix.bval ] && [ -f $prefix.bvec ]
    then
        echo $i
        
        b0=${prefix}_b0.nii.gz
        mask=${prefix}
        
        fslroi $i $b0 0 1
        bet $b0 $mask -m -n       
        
        echo `pwd`/$i >> $IMAGELIST
        echo `pwd`/${prefix}_mask.nii.gz >> $MASKLIST
        echo `pwd`/${prefix}_b0.nii.gz >> $B0LIST
        echo `pwd`/${prefix}_b0-Warped.nii.gz >> $WarpedB0LIST
        echo `pwd`/${prefix}_b0-Warped-mask.nii.gz >> $WarpedMASKLIST
        echo $prefix >> $CASELIST
    fi
done
popd


# ================================


# prediction
export FILTER_METHOD=PYTHON
# ../pipeline/dwi_masking.py -i $IMAGELIST -nproc 5 -f ../model_folder/


# training
../src/registration.py -b0 $B0LIST -mask $MASKLIST -ref ../model_folder/IITmean_b0_256.nii.gz
../src/preprocess_b0.py -i $WarpedB0LIST
../src/preprocess_mask.py -i $WarpedMASKLIST


echo -e "
[COMMON]
save_model_dir = \"$testDataDir/model_folder_test\"
log_dir = \"$testDataDir/log_dir\"

[DATA]
data_dir = \"$prismaDir\"
train_data_file = \"sagittal-traindata-dwi.npy\"
train_label_file = \"sagittal-traindata-mask.npy\"

[TRAINING]
principal_axis = \"sagittal\"
learning_rate = 1e-3
train_batch_size = 4
validation_split = 0.2
num_epochs = 1
shuffle_data = \"True\"
" > $testDataDir/settings.ini


export COMPNET_CONFIG=$testDataDir/settings.ini
../src/train.py

# prediction
cp ../model_folder/IITmean_b0_256.nii.gz model_folder_test/
../pipeline/dwi_masking.py -i $IMAGELIST -nproc 5 -f model_folder_test/


