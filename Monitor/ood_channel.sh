#!/bin/bash
#---------------------------------------------------
python -u /home/ns38942/DeepRx/deep-rx-sim-env/Monitor/main.py \
--gpu_id $1 \
--channel_param 'Channel_Profile' \
--train_dataset_path '../../MonitorDatasetsPkl/training_rx_pilot_for_10_grids_channelprofiles.pkl' \
--test_dataset_path '../../MonitorDatasetsPkl/test_rx_pilot_for_10_grids_channelprofiles.pkl' \
--train \
--test \
--weight_path '/home/ns38942/DeepRx/deep-rx-sim-env/Monitor/results/weights_Channel_Profile.pt' \
> ./results/log-Channel_Profile.out \
2> ./results/log-Channel_Profile.err
