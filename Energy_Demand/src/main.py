import torch 
import numpy as np 
import os
import matplotlib.pyplot as plt 
from arguments import get_args
from neural_network import LSTMModel, FeedForwardModel
from dataset_loader import EnergyDataset 

""" This file consolidates all of the other files in one to perform training and testing
1. We setup the dataset from the dataset_loader.py file
2. We setup the model from the neural_network.py file
3. We define our loss function & optimizer
4. We train, val and test the model
5. We save the model outputs in ./saved_models directory ; where you can find pre-trained weights and output trajectories
"""

args = get_args()

if args.device=='cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:{}'.format(args.device))

def train(dataloader, model, optimizer, loss_fnc):
    model.train()
    train_loss =[]

    for i, data in enumerate(dataloader):
        input, gt = data
        input= input.to(device).to(torch.float32)
        if args.model_name=='ffn':
            input = input.squeeze(2)
        gt = gt.to(device).to(torch.float32).squeeze(2)
        #gt = gt.squeeze(2)
        
        output = model(input)
        loss = loss_fnc(output, gt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        train_loss.append(loss.item())
    
    # return avg loss:
    return np.mean(np.asarray(train_loss))

# test uses the same function as val hence, no need to write an additional test function
def val(dataloader, model, loss_fnc):
    model.eval()
    val_loss =[]
    
    for i, data in enumerate(dataloader):
        input, gt = data
        input= input.to(device).to(torch.float32)
        if args.model_name=='ffn':
            input = input.squeeze(2)
        gt = gt.to(device).to(torch.float32).squeeze(2)
        with torch.no_grad():
            output = model(input)
            loss = loss_fnc(output, gt)

        val_loss.append(loss.item())
    
    # return avg loss:
    return np.mean(np.asarray(val_loss))


def map_predictions(dataloader, model, test_data, test_loss):
    model.eval()
    predictions=[]
    GT = []
    #test_loss=[]
    
    for i, data in enumerate(dataloader):
        input, gt = data
        input= input.to(device).to(torch.float32)
        if args.model_name=='ffn':
            input = input.squeeze(2)
        gt = gt.to(device).to(torch.float32).squeeze(2)

        with torch.no_grad():
            output = model(input)
        predictions.append(output.squeeze(0).cpu().numpy())
        GT.append(gt.squeeze(0).cpu().numpy())
    
    # convert to array;
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(GT)

    # inverse transform
    preds = test_data.scaler.inverse_transform(predictions)
    original = test_data.scaler.inverse_transform(ground_truth)

    #plt.axvline(x=np.arange(0, len(preds)), c='r', linestyles='--')
    plt.plot(np.arange(0, len(preds)), original, label='ground truth (original)')
    plt.plot(np.arange(0, len(preds)), preds, label='predictions')
    plt.suptitle('Energy Demand Prediction, Test MSE Score: {:.5f}'.format(test_loss))
    plt.xlabel('time steps')
    plt.ylabel('Consumption')
    plt.legend()
    plt.savefig('{}/{}.png'.format(args.save_dir, args.model_name))
    plt.show()

def main():
    # create save directory;
    if os.path.exists(args.save_dir)!=True:
        os.makedirs(args.save_dir)

    # get dataset
    train_data = EnergyDataset(args, 'train')
    val_data = EnergyDataset(args, 'val')
    test_data = EnergyDataset(args, 'test')
    
    print('train data samples:', len(train_data))
    print('val data samples:', len(val_data))
    print('test data samples:', len(test_data))

    train_loader= torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle=True)
    val_loader= torch.utils.data.DataLoader(val_data, batch_size = args.batch_size, shuffle=False)
    test_loader= torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle=False)

    # get model
    if args.model_name=='lstm':
        model = LSTMModel(args).to(device)
    elif args.model_name=='ffn':
        model = FeedForwardModel(args).to(device)
    else:
        raise Exception("model type: {} not implemented! please select either 'lstm' or 'ffn'".format(args.model_name))

    # set loss function + optimizer
    loss_function = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    # TRAIN
    if args.train or args.train_test:
        for i in range(args.epochs):
            train_loss = train(train_loader, model, optimizer, loss_function)
            val_loss = val(val_loader, model, loss_function)
            print('Epoch: {}, Train Loss: {:.3f}, Val Loss:{:.3f}'.format(i, train_loss.item(), val_loss.item()))
        
        print('training complete')
        SaveFileName= os.path.join(args.save_dir, '{}_epoch_{}_loss_{:.3f}.tar'.format(args.model_name, i, val_loss.item()))
        torch.save({'model_dict': model.state_dict(),
                    'model_optim': optimizer.state_dict()}, SaveFileName)
        print('model saved to: {}'.format(args.save_dir))        
    # TEST  
    if args.test or args.train_test:
        test_loss = val(test_loader, model, loss_function)
        print('test loss: {}'.format(test_loss.item()))
        print('testing complete')
        map_predictions(test_loader, model, test_data, test_loss)
        print('predictions figure saved to: {}'.format(args.save_dir))

if __name__=='__main__':
    main()
    