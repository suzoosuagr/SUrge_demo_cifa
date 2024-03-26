#!/bin/bash
SIMG=demo_container.simg

singularity run --bind /home/jwang127/jiyang_example/SUrge_Demo_Cifa:/workplace,/home/jwang127/data:/data\
                --workdir /workplace\
                --nv $SIMG <<EOT
python train.py --data_root=/data/public_dataset/pytorch\
            --weights_root=./model_weights/SUrge_Demo_Cifa/

EOT