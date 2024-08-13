import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
# import torchvision
from torchvision import transforms as T
# from torch.autograd import Variable
import albumentations as A
# from torchsummary import summary
# import segmentation_models_pytorch as smp
from tqdm.notebook import tqdm
import os
import time

## Building encoding circuit for a single band
# dev = qml.device("default.qubit", wires = 7)

# @qml.qnode(dev)
def encoding_circ_fqcnn(pixel_val):
    qml.Hadamard(wires = 0)
    qml.Hadamard(wires = 1)
    qml.Hadamard(wires = 2)
    qml.Hadamard(wires = 3)

    qml.RY(pixel_val[0], wires = 4)
    qml.ctrl(qml.CRY, 0)(pixel_val[1], wires = [2,4])
    # qml.CRY(pixel_val[1], wires = [0,4])
    # qml.CRY(pixel_val[2], wires = [1,4])
    qml.ctrl(qml.CRY, 1)(pixel_val[2], wires = [3,4])
    qml.ctrl(qml.CRY, 0)(pixel_val[3], wires = [1,4])
    qml.Barrier()
    # return qml.expval(qml.PauliZ(4))

# print(qml.draw_mpl(encoding_circ_fqcnn)([0.1,0.2,0.3,0.4]))


## Building the Convolution circuit
def u3_matrix(theta, phi, lam):
    return np.array([
        [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
        [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)]
    ])


def convolution_circ_fqcnn(U3_params):
    # U3 = u3_matrix(U3_params[0], U3_params[1], U3_params[2])
    qml.Hadamard(wires=5)
    ## Band 1
    qml.ControlledQubitUnitary(u3_matrix(U3_params[0].tolist()[0],U3_params[0].tolist()[1],U3_params[0].tolist()[2]), control_wires=[4], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[1].tolist()[0],U3_params[1].tolist()[1],U3_params[1].tolist()[2]), control_wires=[4, 0], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[2].tolist()[0],U3_params[2].tolist()[1],U3_params[2].tolist()[2]), control_wires=[4,2], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[3].tolist()[0],U3_params[3].tolist()[1],U3_params[3].tolist()[2]), control_wires=[4,2,0], wires=[6])
    qml.Barrier()
    ## Band 2
    qml.ControlledQubitUnitary(u3_matrix(U3_params[4].tolist()[0],U3_params[4].tolist()[1],U3_params[4].tolist()[2]), control_wires=[4,5], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[5].tolist()[0],U3_params[5].tolist()[1],U3_params[5].tolist()[2]), control_wires=[4,5,0], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[6].tolist()[0],U3_params[6].tolist()[1],U3_params[6].tolist()[2]), control_wires=[4,5,2], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[7].tolist()[0],U3_params[7].tolist()[1],U3_params[7].tolist()[2]), control_wires=[4,5,0,2], wires=[6])
    qml.Barrier()
    ## Band 3

    # return qml.expval(qml.PauliZ(4))

# print(qml.draw_mpl(convolution_circ_fqcnn)([0.2,0.4,0.5,0.6]))

dev_fqcnn = qml.device("lightning.qubit", wires = 7)

@qml.qnode(dev_fqcnn)
def fqcnn_circ(pixel_val, u3_params):
    encoding_circ_fqcnn(pixel_val)
    convolution_circ_fqcnn(u3_params)

    return qml.expval(qml.PauliZ(4))


## Quantum Layer
class FQCNN_layer(nn.Module):
    def __init__(self):
        super(FQCNN_layer, self).__init__()
        self.U3_w = nn.Parameter(torch.randn(8,3, dtype=torch.float64), requires_grad=True)
    
    def apply_conv(self, image):
        conv_feature = []
        _, im_size, _ = image.shape
        for im in image:
            band_features = []
            for j in range(0, im_size, 2):
                for k in range(0, im_size, 2):
                    pixel_val = [im[j, k],im[j, k+1],im[j+1, k],im[j+1, k+1]]
                    pixel_enc = fqcnn_circ(pixel_val, self.U3_w)
                    band_features.append(pixel_enc)
            band_features = torch.stack(band_features).reshape((im_size // 2, im_size // 2))

            conv_feature.append(band_features)
        
        return torch.stack(conv_feature).float()
    
    
    def forward(self, x):
        batch_size, _, im_size, _ = x.shape 
        encoded_img = []
        ## go through each image in the batch
        for im in x:
            ## perform quantum conv operation and extract features
            feature_band = self.apply_conv(im)
            # print(feature_band_1.shape)
            # print(feature_band.shape)
            encoded_img.append(feature_band)
        # encoded_img_reshaped = np.array(self.encoded_img).reshape(batch_size,3,im_size // 2,im_size // 2)
        encoded_img_reshaped = torch.stack(encoded_img).reshape((batch_size,3,im_size // 2,im_size // 2))
        return encoded_img_reshaped




class FQCNN_Hybrid_Model(nn.Module):
    def __init__(self):
        super(FQCNN_Hybrid_Model, self).__init__()
        self.fqcnn_layer_1 = FQCNN_layer()
        # self.fqcnn_layer_2 = FQCNN_layer()
        self.conv_transpose1 = nn.ConvTranspose2d(3, 64, kernel_size=3, stride=1, padding=1) # Output: 64x64x64
        self.relu1 = nn.ReLU()
        # self.conv_transpose2 = nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1)  # Output: 128x128x5
        # self.relu2 = nn.ReLU()
        # self.conv_transpose3 = nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1)  # Output: 256x256x5
        # self.relu3 = nn.ReLU()
        self.upsample = nn.ConvTranspose2d(in_channels=64, out_channels=5, kernel_size=2, stride=2)
        
    def forward(self, x):
        quantum_features_1 = self.fqcnn_layer_1(x)
        # x1 = quantum_features.view(1,3,64,64).float()
        # print(quantum_features.shape)
        # quantum_features_2 = self.fqcnn_layer_2(quantum_features_1)
        x1 = self.conv_transpose1(quantum_features_1)
        x1 = self.relu1(x1)
        # x1 = self.conv_transpose2(x1)
        # x1 = self.relu2(x1)
        x1 = self.upsample(x1)
        return x1


## Building encoding circuit for a single band
# dev = qml.device("default.qubit", wires = 8)


def encoding_circ_mqcnn(pixel_val):
    qml.Hadamard(wires = 0)
    qml.Hadamard(wires = 1)
    qml.Hadamard(wires = 2)
    qml.Hadamard(wires = 3)

    qml.RY(pixel_val[0], wires = 4)
    qml.ctrl(qml.CRY, 0)(pixel_val[1], wires = [2,4])
    # qml.CRY(pixel_val[1], wires = [0,4])
    # qml.CRY(pixel_val[2], wires = [1,4])
    qml.ctrl(qml.CRY, 1)(pixel_val[2], wires = [3,4])
    qml.ctrl(qml.CRY, 0)(pixel_val[3], wires = [1,4])
    qml.Barrier()

    qml.RY(pixel_val[4], wires = 4)
    qml.ctrl(qml.CRY, 0)(pixel_val[5], wires = [3,4])
    # qml.CRY(pixel_val[1], wires = [0,4])
    # qml.CRY(pixel_val[2], wires = [1,4])
    qml.ctrl(qml.CRY, 0)(pixel_val[6], wires = [2,4])
    qml.ctrl(qml.CRY, 2)(pixel_val[7], wires = [3,4])
    qml.Barrier()

    qml.RY(pixel_val[8], wires = 4)
    qml.ctrl(qml.CRY, 2)(pixel_val[9], wires = [3,4])
    # qml.CRY(pixel_val[1], wires = [0,4])
    # qml.CRY(pixel_val[2], wires = [1,4])
    qml.ctrl(qml.CRY, 0)(pixel_val[10], wires = [3,4])
    qml.ctrl(qml.CRY, 1)(pixel_val[11], wires = [3,4])
    qml.Barrier()

    # return qml.expval(qml.PauliZ(4))

# print(qml.draw_mpl(encoding_circ_mqcnn)([0.1,0.2,0.3,0.4]))


## Building the Convolution circuit
def u3_matrix(theta, phi, lam):
    return np.array([
        [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
        [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)]
    ])


def convolution_circ_mqcnn(U3_params):
    # U3 = u3_matrix(U3_params[0], U3_params[1], U3_params[2])
    qml.Hadamard(wires=5)
    ## Band 1
    qml.ControlledQubitUnitary(u3_matrix(U3_params[0].tolist()[0],U3_params[0].tolist()[1],U3_params[0].tolist()[2]), control_wires=[4], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[1].tolist()[0],U3_params[1].tolist()[1],U3_params[1].tolist()[2]), control_wires=[4, 0], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[2].tolist()[0],U3_params[2].tolist()[1],U3_params[2].tolist()[2]), control_wires=[4,2], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[3].tolist()[0],U3_params[3].tolist()[1],U3_params[3].tolist()[2]), control_wires=[4,2,0], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[4].tolist()[0],U3_params[4].tolist()[1],U3_params[4].tolist()[2]), control_wires=[4,5], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[5].tolist()[0],U3_params[5].tolist()[1],U3_params[5].tolist()[2]), control_wires=[4,5,0], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[6].tolist()[0],U3_params[6].tolist()[1],U3_params[6].tolist()[2]), control_wires=[4,5,2], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[7].tolist()[0],U3_params[7].tolist()[1],U3_params[7].tolist()[2]), control_wires=[4,5,0,2], wires=[6])
    qml.Barrier()
    ## Band 2
    qml.ControlledQubitUnitary(u3_matrix(U3_params[8].tolist()[0],U3_params[8].tolist()[1],U3_params[8].tolist()[2]), control_wires=[4], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[9].tolist()[0],U3_params[9].tolist()[1],U3_params[9].tolist()[2]), control_wires=[4, 0], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[10].tolist()[0],U3_params[10].tolist()[1],U3_params[10].tolist()[2]), control_wires=[4,2], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[11].tolist()[0],U3_params[11].tolist()[1],U3_params[11].tolist()[2]), control_wires=[4,2,0], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[12].tolist()[0],U3_params[12].tolist()[1],U3_params[12].tolist()[2]), control_wires=[4,5], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[13].tolist()[0],U3_params[13].tolist()[1],U3_params[13].tolist()[2]), control_wires=[4,5,0], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[14].tolist()[0],U3_params[14].tolist()[1],U3_params[14].tolist()[2]), control_wires=[4,5,2], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[15].tolist()[0],U3_params[15].tolist()[1],U3_params[15].tolist()[2]), control_wires=[4,5,0,2], wires=[6])
    qml.Barrier()
    ## Band 3
    qml.ControlledQubitUnitary(u3_matrix(U3_params[16].tolist()[0],U3_params[16].tolist()[1],U3_params[16].tolist()[2]), control_wires=[4], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[17].tolist()[0],U3_params[17].tolist()[1],U3_params[17].tolist()[2]), control_wires=[4, 0], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[18].tolist()[0],U3_params[18].tolist()[1],U3_params[18].tolist()[2]), control_wires=[4,2], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[19].tolist()[0],U3_params[19].tolist()[1],U3_params[19].tolist()[2]), control_wires=[4,2,0], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[20].tolist()[0],U3_params[20].tolist()[1],U3_params[20].tolist()[2]), control_wires=[4,5], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[21].tolist()[0],U3_params[21].tolist()[1],U3_params[21].tolist()[2]), control_wires=[4,5,0], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[22].tolist()[0],U3_params[22].tolist()[1],U3_params[22].tolist()[2]), control_wires=[4,5,2], wires=[6])
    qml.ControlledQubitUnitary(u3_matrix(U3_params[23].tolist()[0],U3_params[23].tolist()[1],U3_params[23].tolist()[2]), control_wires=[4,5,0,2], wires=[6])
    qml.Barrier()


    # return qml.expval(qml.PauliZ(4))

# print(qml.draw_mpl(convolution_circ_mqcnn)([0.2,0.4,0.5]))


dev_mqcnn = qml.device("lightning.qubit", wires = 7)

@qml.qnode(dev_mqcnn)
def mqcnn_circ(pixel_val, u3_params):
    encoding_circ_mqcnn(pixel_val)
    convolution_circ_mqcnn(u3_params)

    return qml.expval(qml.PauliZ(4))

# print(qml.draw_mpl(mqcnn_circ)([0.1,0.2,0.3,0.4],[0.2,0.4,0.5]))


class MQCNN_layer(nn.Module):
    def __init__(self):
        super(MQCNN_layer, self).__init__()
        self.U3_w = nn.Parameter(torch.randn(24,3, dtype = torch.float64), requires_grad = True)

    

    def apply_conv(self, im):
        conv_feature = []

        _, im_size, _ = im.shape
        for j in range(0, im_size,2):
            for k in range(0, im_size, 2):
                pixel_val = [im[0, j, k],im[0, j, k+1],im[0, j+1, k],im[0, j+1, k+1],
                             im[1, j, k],im[1, j, k+1],im[1, j+1, k],im[1, j+1, k+1],
                             im[2, j, k],im[2, j, k+1],im[2, j+1, k],im[2, j+1, k+1]]
                pixel_enc = mqcnn_circ(pixel_val, self.U3_w)
                conv_feature.append(pixel_enc)
            
        
        conv_feature = torch.stack(conv_feature).reshape((im_size//2, im_size//2))

        return conv_feature.float()

    def forward(self, x):
        batch_size, _, im_size, _ = x.shape

        encoded_img = []

        for im in x:
            feature_band = self.apply_conv(im)

            encoded_img.append(feature_band)
        
        encoded_img_reshaped = torch.stack(encoded_img).reshape((batch_size,1, im_size // 2, im_size // 2))

        return encoded_img_reshaped
    



class MQCNN_Hybrid_Model(nn.Module):
    def __init__(self):
        super(MQCNN_Hybrid_Model, self).__init__()
        self.mqcnn_layer = MQCNN_layer()
        self.conv_transpose1 = nn.ConvTranspose2d(1, 128, kernel_size=3, stride=1, padding=1) # Output: 64x64x64
        self.relu1 = nn.ReLU()
        # self.conv_transpose2 = nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1)  # Output: 128x128x5
        # self.relu2 = nn.ReLU()
        # self.conv_transpose3 = nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1)  # Output: 256x256x5
        # self.relu3 = nn.ReLU()
        self.upsample = nn.ConvTranspose2d(in_channels=128, out_channels=5, kernel_size=2, stride=2)
        
    def forward(self, x):
        quantum_features = self.mqcnn_layer(x)
        # x1 = quantum_features.view(1,3,64,64).float()
        # print(quantum_features.shape)
        x1 = self.conv_transpose1(quantum_features)
        x1 = self.relu1(x1)
        # x1 = self.conv_transpose2(x1)
        # x1 = self.relu2(x1)
        x1 = self.upsample(x1)
        return x1



dev_coqcnn = qml.device("lightning.qubit", wires = 5)
def X_theta_gate(theta):
    # Define the matrix elements
    i = 1j  # Imaginary unit
    pi = np.pi  # Pi constant
    exp_i_theta_2 = np.exp(i * theta / 2)  # e^(iθ/2)
    exp_minus_i_theta_2 = np.exp(-i * theta / 2)  # e^(-iθ/2)
    cos_pi_2 = np.cos(pi / 2)  # cos(π/2)
    sin_pi_2 = np.sin(pi / 2)  # sin(π/2)

    # Construct the matrix
    matrix = np.array([
        [exp_i_theta_2 * cos_pi_2, -i * exp_i_theta_2 * sin_pi_2],
        [-i * exp_minus_i_theta_2 * sin_pi_2, exp_minus_i_theta_2 * cos_pi_2]
    ], dtype=complex)

    return matrix
@qml.qnode(dev_coqcnn)
def co_cirq(pixel_val, thetas, phis):
    qml.Hadamard(wires=0)
    ## Band 1
    qml.RX(pixel_val[0], wires=1)
    qml.RX(pixel_val[1], wires=2)
    qml.RX(pixel_val[2], wires=3)
    qml.RX(pixel_val[3], wires=4)
    # encoding_circ(pixel_val)
    # qml.QubitUnitary(U,wires=[1,2,3,4])
    qml.ControlledQubitUnitary(X_theta_gate(thetas.tolist()[0]),control_wires=1, wires=2)
    # qml.CNOT(wires=[1, 2])
    qml.ControlledQubitUnitary(X_theta_gate(thetas.tolist()[1]),control_wires=2, wires=3)
    # qml.CNOT(wires=[2, 3])
    qml.ControlledQubitUnitary(X_theta_gate(thetas.tolist()[2]),control_wires=3, wires=4)
    # qml.CNOT(wires=[3, 4])
    qml.ControlledQubitUnitary(X_theta_gate(thetas.tolist()[3]),control_wires=4, wires=1)
    qml.CPhase(phis.tolist()[0], wires=[0,1])
    qml.Barrier()

    ## Band 2
    qml.RX(pixel_val[4], wires=1)
    qml.RX(pixel_val[5], wires=2)
    qml.RX(pixel_val[6], wires=3)
    qml.RX(pixel_val[7], wires=4)
    # encoding_circ(pixel_val)
    # qml.QubitUnitary(U,wires=[1,2,3,4])
    qml.ControlledQubitUnitary(X_theta_gate(thetas.tolist()[4]),control_wires=1, wires=2)
    # qml.CNOT(wires=[1, 2]
    qml.ControlledQubitUnitary(X_theta_gate(thetas.tolist()[5]),control_wires=2, wires=3)
    # qml.CNOT(wires=[2, 3])
    qml.ControlledQubitUnitary(X_theta_gate(thetas.tolist()[6]),control_wires=3, wires=4)
    # qml.CNOT(wires=[3, 4])
    qml.ControlledQubitUnitary(X_theta_gate(thetas.tolist()[7]),control_wires=4, wires=1)
    qml.CPhase(phis.tolist()[1], wires=[0,1])
    qml.Barrier()
    

    ## Band 3
    qml.RX(pixel_val[8], wires=1)
    qml.RX(pixel_val[9], wires=2)
    qml.RX(pixel_val[10], wires=3)
    qml.RX(pixel_val[11], wires=4)
    # encoding_circ(pixel_val)
    # qml.QubitUnitary(U,wires=[1,2,3,4])
    qml.ControlledQubitUnitary(X_theta_gate(thetas.tolist()[8]),control_wires=1, wires=2)
    # qml.CNOT(wires=[1, 2])
    qml.ControlledQubitUnitary(X_theta_gate(thetas.tolist()[9]),control_wires=2, wires=3)
    # qml.CNOT(wires=[2, 3])
    qml.ControlledQubitUnitary(X_theta_gate(thetas.tolist()[10]),control_wires=3, wires=4)
    # qml.CNOT(wires=[3, 4])
    qml.ControlledQubitUnitary(X_theta_gate(thetas.tolist()[11]),control_wires=4, wires=1)
    qml.CPhase(phis.tolist()[2], wires=[0,1])
    qml.Barrier()
    
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(wires=0))


# print(qml.draw_mpl(co_cirq)([0.2,0.3,0.4,0.5, 0.2, 0.3, 0.4,0.5, 0.2, 0.3, 0.4,0.5],
#                             [0.2,0.3,0.4,0.5,0.2,0.3,0.4,0.5,0.2,0.3,0.4,0.5],
#                             [0.2]))



class COCQCNN_layer(nn.Module):
    def __init__(self):
        super(COCQCNN_layer, self).__init__()
        self.thetas = nn.Parameter(torch.randn(12, dtype = torch.float64), requires_grad = True)
        self.phis = nn.Parameter(torch.randn(3, dtype=torch.float64), requires_grad = True)
    
    def apply_conv(self, im):
        conv_feature = []
        _, im_size, _ = im.shape
        band_features = []
        for j in range(0, im_size, 2):
            for k in range(0, im_size, 2):
                pixel_val = [im[0,j, k],im[0,j, k+1],im[0,j+1, k],im[0,j+1, k+1],
                                im[1,j, k],im[1,j, k+1],im[1,j+1, k],im[1,j+1, k+1],
                                im[2,j, k],im[2,j, k+1],im[2,j+1, k],im[2,j+1, k+1]]
                pixel_enc = co_cirq(pixel_val, self.thetas, self.phis)
                band_features.append(pixel_enc)
        band_features = torch.stack(band_features).reshape((im_size // 2, im_size // 2))

        conv_feature.append(band_features)
        
        return torch.stack(conv_feature).float()

    def forward(self, x):
        batch_size, _, im_size, _ = x.shape 
        encoded_img = []
        ## go through each image in the batch
        for im in x:
            ## perform quantum conv operation and extract features
            feature_band = self.apply_conv(im)
            # print(feature_band_1.shape)
            # print(feature_band.shape)
            encoded_img.append(feature_band)
        # encoded_img_reshaped = np.array(self.encoded_img).reshape(batch_size,3,im_size // 2,im_size // 2)
        encoded_img_reshaped = torch.stack(encoded_img).reshape((batch_size,1,im_size // 2,im_size // 2))
        return encoded_img_reshaped



class Hybrid_COQCNN_Model(nn.Module):
    def __init__(self):
        super(Hybrid_COQCNN_Model, self).__init__()
        self.coqcnn_layer = COCQCNN_layer()
        # self.mqcnn_layer_2 = COCQCNN_layer()
        self.conv_transpose1 = nn.ConvTranspose2d(1, 128, kernel_size=3, stride=1, padding=1) # Output: 64x64x64
        self.relu1 = nn.ReLU()
        # self.conv_transpose2 = nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1)  # Output: 128x128x5
        # self.relu2 = nn.ReLU()
        # self.conv_transpose3 = nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1)  # Output: 256x256x5
        # self.relu3 = nn.ReLU()
        self.upsample = nn.ConvTranspose2d(in_channels=128, out_channels=5, kernel_size=2, stride=2)
        
    def forward(self, x):
        quantum_features = self.coqcnn_layer(x)
        x1 = self.conv_transpose1(quantum_features)
        x1 = self.relu1(x1)
        x1 = self.upsample(x1)
        return x1
