from model import FCN
from data import get_training_data
from train import train
import torch

if __name__ == "__main__":
    layers = [2, 20, 20, 20, 20, 1]
    model = FCN(layers).cpu()

    X_u_train, u_train, X_f_train = get_training_data()
    
    train(model, X_u_train, u_train, X_f_train, nu=0.01 / torch.pi)
