#!/bin/bash
#---------------------------------------------------
python -u /home/ns38942/DeepRx/deep-rx-sim-env/Monitor/main.py \
--gpu_id $1 \
--channel_param 'Tx_Speed' \
--train_dataset_path '../../MonitorDatasetsPkl/training_rx_pilot_for_10_grids_speed.pkl' \
--test_dataset_path '../../MonitorDatasetsPkl/test_rx_pilot_for_10_grids_speed.pkl' \
--train \
--test \
--weight_path '/home/ns38942/DeepRx/deep-rx-sim-env/Monitor/results/weights_Tx_Speed.pt' \
> ./results/log-Tx_Speed.out \
2> ./results/log-Tx_Speed.err
