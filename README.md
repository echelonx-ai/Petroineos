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
 Once all the requirements have been installed, you can run the code. There are multiple arguments that can be edited either in the `./Energy_Demand/arguments.py` file, or simply via the terminal by using the relevant flag. 
 
 We experiment with two models: `LSTM` and `Multi Layer Perceptron`. The type of model you want to train / test, can be selected by editing the model name arg i.e. `--model_name $model_name` where `$model_name` is a string option `"lstm"` or `"ffn"` to select the LSTM or Multi Layer Perceptron model respectively.

#### Training & Testing:
``` python
python ./Energy_Demand/src/main.py --train_test --model_name $model_name --batch_size $batch_size --data_path $data_path
```
Where the flag `train_test` specifies conducting all training, validation and testing. `--data_path` flag, requires the file path to the provided dataset `energy.dat`.

#### Training only:

#### Testing only:

### Data Exploration

### Forecasting Results

## Coin Machine

### Instructions