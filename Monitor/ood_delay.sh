#!/bin/bash
#---------------------------------------------------
python -u /home/ns38942/DeepRx/deep-rx-sim-env/Monitor/main.py \
--gpu_id $1 \
--channel_param 'Delay_Spread' \
--train_dataset_path '../../MonitorDatasetsPkl/training_rx_pilot_for_10_grids_delay.pkl' \
--test_dataset_path '../../MonitorDatasetsPkl/test_rx_pilot_for_10_grids_delay.pkl' \
--train \
--test \
--weight_path '/home/ns38942/DeepRx/deep-rx-sim-env/Monitor/results/weights_Delay_Spread.pt' \
> ./results/log-Delay_Spread.out \
2> ./results/log-Delay_Spread.err
