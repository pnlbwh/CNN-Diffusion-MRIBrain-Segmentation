### Tests

[pipeline_test.sh](./pipeline_test.sh) is a convenient testing infrastructure.
In addition to the prerequisites mentioned in [instructions](../docs/README.md#installation),
it is depdendent on the following executables:

* dcm2niix
* fslroi
* bet

We recommend only putting the above executables in `PATH` without sourcing extenal environments
such as FSL's. For example--`fslroi`, `bet` could be put in `PATH` as:

> export PATH=/path/to/fsl-6.0.7/share/fsl/bin:$PATH

In addition, you need to set:

> export FSLOUTPUTTYPE=NIFTI_GZ

Moreover, you can put an `exit` statement after any line between 70-81 to run one or more tests:
https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation/blob/a81ef7e939714f88b67c0a6e84a0ff6db7004622/tests/pipeline_test.sh#L70-L81

