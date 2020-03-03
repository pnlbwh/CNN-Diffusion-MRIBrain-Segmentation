#!/usr/bin/env python
from compnet import train_model
from settings import Settings


if __name__ == '__main__':
    
    settings = Settings()

    data_params = settings['DATA']
    train_params = settings['TRAINING']
    common_params = settings['COMMON']

    train_model(data_params, train_params, common_params)
