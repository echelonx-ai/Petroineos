import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--input_sequence_length', type=int, default=5, help="number of points fed as input into the model")
    parser.add_argument('--output_sequence_length', type=int, default=1, help="number of points outputted by the model")
    parser.add_argument('--data_path', type=str, default='/home/Petroineos/Energy_Demand/energy.dat', help="dir path of where the data is stored")

    # optimizer arguments
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--lr', type=float, default=5e-5, help="optimizer learning rate")
    parser.add_argument('--epochs', type=int, default=160, help="total training epochs")
    
    # training arguments
    parser.add_argument('--device', type=int, default=1, help="cuda device to use for training, otherwise use 'cpu' if no cuda available")
    parser.add_argument('--train', action='store_true', help="flag for only training")
    parser.add_argument('--test', action='store_true', help="flag for only testing")
    parser.add_argument('--train_test', action='store_true', help="flag for conducting both training and testing in one go")
    parser.add_argument('--save_dir', type=str, default='./saved_models', help="directory for saving trained models")

    # model arguments
    parser.add_argument('--model_name', type=str, default='lstm', help="the type of model you want to train; ffn or lstm")
    parser.add_argument('--num_layers', type=int, default=1, help="number of mlp layers in lstm")
    parser.add_argument('--hidden_size', type=int, default=128, help="hidden layer neuron count")
    parser.add_argument('--input_feature_size', type=int, default=1, help="input features")



    args = parser.parse_args()
    return args