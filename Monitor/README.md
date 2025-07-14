# Monitor
This code is for training and testing the Monitor component in VERITAS system.

## Downloadables
- Dataset pickles: Download dataset files [here](https://utexas-my.sharepoint.com/:f:/g/personal/ns38942_eid_utexas_edu/EhXQ1NEipmdFtSx-ptydVmoBJ3md3aUkLzB_tCnKu9Xb9w?e=F79xjt).
- Trained models: Download trained weights [here](https://utexas-my.sharepoint.com/:f:/g/personal/ns38942_eid_utexas_edu/EjKqeyTTgYhLu1VFJCrpynUBQdqd7-Z6ybUochsQ8eAo0A?e=SVeKNH).

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
python -u /home/ns38942/VERITAS/Monitor/main.py \
--gpu_id $1 \
--channel_param 'Channel_Profile' \
--train_dataset_path '../MonitorDatasetsPkl/training_rx_pilot_for_10_grids_channelprofiles.pkl' \
--test_dataset_path '../MonitorDatasetsPkl/test_rx_pilot_for_10_grids_channelprofiles.pkl' \
--train \
--test \
--weight_path '/home/ns38942/VERITAS/Monitor/results/weights_Channel_Profile.pt' \
> ./results/log-Channel_Profile.out \
2> ./results/log-Channel_Profile.err
```

- This bash file runs the `main.py` script, provides inputs to it, and writes output logs to the `.out` and `.err` files in the `result` folder.
- Additionally, if `train` is activated, the trained weights will be saved in `results` folder.
- Also, if `test` is activated, the generated output features are saved in `results` folder.

**To start Training** 
- Download the dataset using the provided links.
- Change the dataset paths in the `bash` file according to your own system.
- `cd` into the `Monitor` folder.
- provide GPU id through the command line, and run:
```
./ood_channel.sh 0
```

- If you keep `--test` variable in the `bash` file, test will also be done after training.
- After training and test, the trained weights and predictions will be saved in the `results` folder.

**To do test without training**
- Download the trained weights using the provided links.
- Remove the `--train` variable in the `bash` file, and keep only `--test`.
- Change the datasets and weight paths in the `bash` file according to your own system.
- `cd` into the `Monitor` folder.
- provide GPU id through the command line, and run:
```
./ood_channel.sh 0
```

### Running the OOD detection algorithm
After training and test for each channel parameter is done, OOD detection algorithm and plotting the results can be done through the scripts provided the 3 `.ipynb` files:
- `OOD_detection_and_plot_Channel_Profile.ipynb`
- `OOD_detection_and_plot_Tx_Speed.ipynb`
- `OOD_detection_and_plot_Delay_Spread.ipynb`
 
To use these jupyter notebook scripts the path in the second cell needs to be changed according to your system.
```
pred_pkl_path = '/home/ns38942/VERITAS/Monitor/results/preds_Channel_Profile.pkl'
```
- Our `.pkl` prediction files are provided [here](https://utexas-my.sharepoint.com/:f:/g/personal/ns38942_eid_utexas_edu/EjKqeyTTgYhLu1VFJCrpynUBQdqd7-Z6ybUochsQ8eAo0A?e=SVeKNH) for re-plotting the paper results without having to do the training and test processes.

