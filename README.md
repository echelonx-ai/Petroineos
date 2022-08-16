# Petroineos

This readme gives an overview of the two subtasks completed:
1. Energy Demand
2. Coin Machine

Each subtask has it's own respective folder by the same name, that contains the code written for completing this challenge.

## Energy Demand
All code for Energy Demand is in the following directory: `./Energy_Demand`
### Setup
To run the code for energy demand, we use deep learning modelling librariry [PyTorch](https://pytorch.org/) hence install the necessary requirements to run the file (ideally in virtual env):
``` python
pip install -r ./Energy_Demand/requirements.txt
```
### Instructions
Once all the requirements have been installed, you can run the code. The code for Energy Demand comprises of the following files:

1. `dataset_loader.py`: Which defines all the data pre-processing steps wrapped in a class, to create a dataset for training a model
2. `neural_network.py`: Defines the code for the 2 types of model we experiment with. I particularly experiment with two models: `LSTM` and `Multi Layer Perceptron`
3. `arguments.py`: Defines all the hyper parameters and other args to run the code
4. `main.py`: Defines functions for training, validation and testing the model and saving the weights

There are multiple arguments that can be edited either in the `./Energy_Demand/arguments.py` file, or simply via the terminal by using the relevant flag For example the type of model you want to train / test, can be selected by editing the model name arg i.e. `--model_name $model_name` where `$model_name` is a string option `"lstm"` or `"ffn"` to select the LSTM or Multi Layer Perceptron model respectively.

#### Training & Testing:
``` python
python ./Energy_Demand/src/main.py --train_test --device $device --model_name $model_name --batch_size $batch_size --data_path $data_path
```
Where the flag `train_test` specifies conducting all training, validation and testing. `--data_path` flag, requires the file path to the provided dataset `energy.dat`. `--device` flag allows you to select if you'd like to use the CPU or the GPU to train. If CPU is preferred the the arg should be a string `"cpu"` otherwise an int stating the GPU device id i.e. `1` if you'd like to use the primary GPU.

#### Training only:
If you'd like to perform training and validation only (i.e. no testing), then simply use the `--train` flag. The rest of the args can stay the same i.e.
``` python
python ./Energy_Demand/src/main.py --train ...
```
#### Testing only:
If you'd like to perform testing only, then simply use the `--test` flag. The rest of the args can stay the same i.e.
``` python
python ./Energy_Demand/src/main.py --test ...
```
### Data Exploration

### Forecasting Results

## Coin Machine

### Instructions