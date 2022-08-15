import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_sequence_length', type=int, default=5, help="number of points fed as input into the model")
    parser.add_argument('--output_sequence_length', type=int, default=1, help="number of points outputted by the model")
    parser.add_argument('--data_path', type=str, default='./home/sam37avhvaptuka451/Documents/Contracts/Petroineos/Energy_Demand/energy.dat', help="dir path of where the data is stored")


    args = parser.parse_args()
    return args