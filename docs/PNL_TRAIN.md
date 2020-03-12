![](pnl-bwh-hms.png)

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.3665739.svg)](https://doi.org/10.5281/zenodo.3665739) [![Python](https://img.shields.io/badge/Python-3.6-green.svg)]() [![Platform](https://img.shields.io/badge/Platform-linux--64%20%7C%20osx--64-orange.svg)]()

*CNN-Diffusion-MRIBrain-Segmentation* repository is developed by Senthil Palanivelu, Suheyla Cetin Karayumak, Tashrif Billah, Sylvain Bouix, and Yogesh Rathi, 
Brigham and Women's Hospital (Harvard Medical School).

# Training CNN at PNL

Training the CNN on new data requires a reasonably powerful GPU machine. Two such 
machines are available at PNL: `pnl-oracle` and `pnl-maxwell`. You can ssh into them 
as follows:

    
    ssh pnl-oracle
    ssh pnl-maxwell
    
    
You may need to append `.bwh.harvard.edu` suffix at the end of the above hostnames.

Then, source the following environment that makes use of `tensorflow-gpu`:


    source /rfanfs/pnl-zorro/software/pnlpipe3/CNN-Diffusion-MRIBrain-Segmentation/train_env.sh
    
    
Finally, follow the instruction from [README.md#training](https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation#training) to perform training.


