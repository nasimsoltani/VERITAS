# Monitor
This code is for training and testing the Monitor component in VERITAS system.

## Downloadables
- Dataset pickles: 
- Trained models:

## Code Structure
There are 6 `.py` files that collectively train the Monitor NN on the training set, test it on the test set, and extract output features.

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
