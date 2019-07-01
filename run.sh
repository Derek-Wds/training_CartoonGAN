#!/bin/bash

python main.py --batch_size=32 \
--epochs=500 \
--gpu_num=2 \
--image_channels=3 \
--image_size=256 \
--init_epoch=30 \
--lr=0.0002