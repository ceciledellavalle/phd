# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# General import
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import cv2 as cv
import os
from PIL import Image
import matplotlib.pyplot as plt
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# Local import
from FBResNet.myfunc import Physics
from FBResNet.myfunc import MyMatmul
from FBResNet.model import MyModel
from FBResNet.bartlett import Test_cuda

        
class FBRestNet(nn.Module):
    """
    Includes the main training and testing methods of iRestNet.
    Attributes
    ----------
        im_size        (numpy array): image size
        path_test              (str): path to the folder containing the test sets
        path_train             (str): path to the training set folder 
    """
    def __init__(self, nb_blocks=200, experimentation, noise = 0.1,\
                 folder, im_set="Set1",batch_size=[50,5],\
                 lr_i=1e-2, nb_epochs=[10,1]):
        """
        Parameters
        ----------
           
        """
        super(FBRestNet, self).__init__()   
        # physical information
        self.physics    = experimentation
        self.noise      = noise
        # training information
        self.lr_i       = lr
        self.nb_epochs  = nb_epochs[0]
        self.freq_val   = nb_epochs[1]
        self.nb_blocks  = nb_blocks
        self.nsamples   = batch_size[0]
        self.train_size = batch_size[1] # training set 
        self.val_size   = 1            # and validation set/test set 
        self.im_set     = im_set
        self.loss_fn    = torch.nn.MSELoss(size_average=True)
        # saving info
        self.path       = folder
        # model creation
        self.model      = MyModel(self.physics,nL=self.nb_blocks)

    
    def CreateDataSet(self,save='yes'):
        """
        Creates the dataset from an image basis, rescale, compute transformation and noise.
        Construct the appropriate loader for the training and validation sets.
        Parameters
        ----------
            save       (str) : 'yes' if the data are saved for reloading.
        Returns
        -------
            train_loader
            val_loader
        """
        #
        Test_cuda
        self.device   =
        self.dtype    = 
        # Recuperation des donnees
        nx            = self.physics.nx
        noise         = self.noise
        nsample       = self.nsamples
        im_set        = self.im_set
        # Initialisation
        color         = ('b','g','r')
        #
        liste_l_trsf  = []
        liste_tT_trsf = []
        #
        save_lisse    = []
        save_l_trsf   = []
        save_blurred  = []
        save_tT_trsf  = []
        # Upload Data
        # path : './MyResNet/Datasets/Images/'
        for folder, subfolders, filenames in os.walk(self.path+'Images/'+im_set+'/'): 
            for img in filenames: 
                item       = folder+img
                img_cv     = cv.imread(item,cv.IMREAD_COLOR)
                for i,col in enumerate(color):
                    # Etape 1 : obtenir l'histogramme lisse des couleurs images
                    histr  = cv.calcHist([img_cv],[i],None,[256],[0,256]).squeeze()
                    # Savitzky-Golay
                    y      = savgol_filter(histr, 21, 5)
                    # interpolation pour nx points
                    x      = np.linspace(0,1,256, endpoint=True)
                    xp     = np.linspace(0,1,nx,endpoint=True)
                    f      = interp1d(x,y)
                    yp     = f(xp)
                    # normalisation
                    ncrop         = nx//20
                    yp[:ncrop]    = 0
                    yp[nx-ncrop:] = 0
                    yp[yp<0]      = 0
                    x_true        = yp/np.amax(yp)
                    # reshaping in channelxm
                    x_true        = x_true.reshape(1,-1)
                    # save
                    save_lisse.append(x_true.squeeze())
                    # Etape 2 : passage dans la base de T^*T
                    x_true_trsf = test.BasisChange(x_true)
                    # save
                    liste_l_trsf.append(x_true_trsf)
                    save_l_trsf.append( x_true_trsf.squeeze())
                    #  Etape 3 : obtenir les images bruitees par l' operateur d' ordre a
                    # transform and add noise
                    x_blurred   = test.Compute(x_true) 
                    # Etape 5 : Bruitage 
                    noise_vect  = np.random.randn(nx).reshape(1,-1)
                    noise_vect  = noise_vect/np.linalg.norm(noise_vect)
                    x_blurred  += noise*np.linalg.norm(x_blurred)*noise_vect
                    # save
                    save_blurred.append(x_blurred.squeeze())
                    # Etape 6 : compute adjoint in the cos basis
                    tTx_blurred = test.ComputeAdjoint(x_blurred)
                    # and save
                    liste_tT_trsf.append(tTx_blurred)
                    save_tT_trsf.append( tTx_blurred.squeeze())
        # Export data in .csv
        if save =='yes':
            np.savetxt(self.path+'Signals/data_lisse.csv',      save_lisse,   delimiter=', ', fmt='%12.8f')
            np.savetxt(self.path+'Signals/data_lisse_trsf.csv', save_l_trsf,  delimiter=', ', fmt='%12.8f')
            np.savetxt(self.path+'Signals/data_blurred.csv',    save_blurred, delimiter=', ', fmt='%12.8f')
            np.savetxt(self.path+'Signals/data_tTblurred.csv',  save_tT_trsf, delimiter=', ', fmt='%12.8f')
        # Tensor completion
        x_tensor = torch.FloatTensor(liste_l_trsf) # signal in cos basis
        y_tensor = torch.FloatTensor(liste_tT_trsf)# blurred and noisy signal in element basis
        #
        dataset = TensorDataset(y_tensor[:nsample], x_tensor[:nsample])
        l       = len(dataset)
        ratio   = 2*l//3
        train_dataset, val_dataset = random_split(dataset, [ratio, l-ratio])
        #
        train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)
        #
        return train_loader, val_loader

    def LoadDataSet(folder):
        """
        Dreates the appropriate loader for the training and validation sets
        when the dataset is already created.
        """
        #
        nsample = self.nsample
        #
        dfl     = pd.read_csv(folder+'/'+'data_lisse_trsf.csv', sep=',',header=None)
        dfb    = pd.read_csv(folder+'/'+'data_tTblurred.csv', sep=',',header=None)
        _,m     = dfl.shape
        _,nx    = dfb.shape
        #
        x_tensor = torch.FloatTensor(dfl.values[:nsample]).view(-1,1,m)
        y_tensor = torch.FloatTensor(dfb.values[:nsample]).view(-1,1,nx)
        #
        dataset = TensorDataset(y_tensor, x_tensor)
        l = len(dataset)
        ratio = 2*l//3
        train_dataset, val_dataset = random_split(dataset, [ratio, l-ratio])
        #
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)
        #
        return train_loader, val_loader
    
    def train(self,train_set,val_set,test_lipschitz=False,save_model=False):
        """
        Trains iRestNet.
        """      
        # to store results
        nb_epochs    = self.nb_epochs
        nb_val       = self.nb_epochs//self.freq_val
        loss_train   =  np.zeros(nb_epochs)
        loss_val     =  np.zeros(nb_val)
        lip_cste     =  np.zeros(nb_val)
        # defines the optimizer
        lr_i        = self.lr_i
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.model.parameters()),lr=self.lr_i)
        
        #==========================================================================================================
        # trains for several epochs
        for epoch in range(0,self.nb_epochs): 
            # sets training mode
            self.model.train()
            # modifies learning rate
            if epoch>0:
                lr_i      = lr_i*0.9 
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=lr_i)
            # TRAINING
            # goes through all minibatches
            for i,minibatch in enumerate(train_set):
                [y, x] = minibatch    # get the minibatch
                x_init    = Variable(y,requires_grad=False)
                x_true    = Variable(x,requires_grad=False)
                x_pred    = self.model(x_init,x_init) 
                    
                # Computes and prints loss
                loss               = loss_fn(x_pred, x_true)
                loss_train[epoch] += torch.Tensor.item(loss)
                    
                # sets the gradients to zero, performs a backward pass, and updates the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # normalisation
            loss_train[epoch] = loss_train[epoch]/i
            #
            # VALIDATION AND STATS
            if epoch%self.freq_val==0:
                # Validation
                with torch.no_grad():
                # tests on validation set
                    self.model.eval()      # evaluation mode
                    for i,minibatch in enumerate(val_set):
                        [y, x] = minibatch            # gets the minibatch
                        x_true  = Variable(x,requires_grad=False)
                        x_init  = Variable(y,requires_grad=False)
                        x_pred  = self.model(x_init,x_init).detach()
                    
                        # computes loss on validation set
                        loss_val[epoch] += torch.Tensor.item(self.loss_fn(x_pred, x_true))
                    # normalisation
                    loss_val[epoch] = loss_val[epoch]/i
                # print stat
                print("epoch : ", epoch," ----- ","validation : ",loss_val[epoch])
                # Test Lipschitz
                lip_cste = self.model.Lifschitz()
                
            
        #==========================================================================================================
        # training is finished
        print('-----------------------------------------------------------------')
        print('Training is done.')
        print('-----------------------------------------------------------------')
        
        # Plots
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(np.linspace(0,1,nb_epochs),loss_train,label = 'train')
        ax1.plot(np.linspace(0,1,nb_val),loss_train,label = 'val')
        ax2.plot(np.linspace(0,1,nb_val),lip_cste,'r-')
        ax2.title("Lipschitz constant")
        plt.legend()
        plt.show()
        
        # Save model
        if save_model:
            torch.save(self.model.state_dict(), self.path+'Trainings/param{}{}.pt'.format(self.a,self.p))
    
    def test(self, dataset):    
        """
        Parameters
        ----------
            dataset        (str): name of the test set
        """
        with torch.no_grad():
            self.model.eval()
        # to do
        
