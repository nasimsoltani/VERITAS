# Monitor
This code is for training and testing the Monitor component in VERITAS system.

## Downloadables
- Dataset pickles: https://utexas-my.sharepoint.com/:f:/g/personal/ns38942_eid_utexas_edu/EhXQ1NEipmdFtSx-ptydVmoBJ3md3aUkLzB_tCnKu9Xb9w?e=F79xjt
- Trained models: https://utexas-my.sharepoint.com/:f:/g/personal/ns38942_eid_utexas_edu/EjKqeyTTgYhLu1VFJCrpynUBQdqd7-Z6ybUochsQ8eAo0A?e=SVeKNH

## Code Structure
- There are 6 `.py` files that collectively train the Monitor NN on the training set, test it on the test set, and extract output features.
- There are 3 `.sh` files for running the `.py` files and perform training and test.
- There are 3 `.ipynb` files for running the OOD detection algorithm and plotting results.

### Training the Testing the Monitor NN

```
main.py
monitor_train_dataset_reader.py
monitor_dataset.py
monitor_models.py
monitor_training.py
monitor_test_model.py
```
Among these, `main.py` is the top file that calls different classes and functions in other modules. `main.py` takes a few input arguments through `argparse` as follows:


- `--gpu_id`: An integer indicating the gpu you want to use on your system.
- `--train_dataset_path`: Path to the `.pkl` file that contains training dataset.
- `--test_dataset_path`: Path to the `.pkl` file that contains test dataset.
- `--channel_param`: The channel parameter you want to detect a change in. Options are `Channel_Profile`, `Tx_Speed`, and `Delay_Spread`.
- `--train`: A `store_true` variable that determines whether we want to train the model or not. 
- `--test`: A `store_true` variable that determines whether we want to test the model or not. 
- `--weight_path`: Path to trained weights. This is only necessary if we are only testing a model without first training it.

These input parameters are provided to `main.py` using a bash file with `.sh` extension.
Example bash file:
```
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
```
This bash file runs the `main.py` script, provides inputs to it, and writes output logs to the `.out` and `.err` files in the `result` folder.
Additionally, if `train` is activated, the trained weights will be saved in `results` folder.
Also, if `test` is activated, the generated output features are saved in `results` folder.


