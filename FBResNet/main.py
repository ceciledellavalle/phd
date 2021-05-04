# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# General import
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from torch.autograd import Variable
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
from FBResNet.posttreat import Export_Data

        
class FBRestNet(nn.Module):
    """
    Includes the main training and testing methods of iRestNet.
    Attributes
    ----------
        im_size        (numpy array): image size
        path_test              (str): path to the folder containing the test sets
        path_train             (str): path to the training set folder 
    """
#========================================================================================================
#========================================================================================================
    def __init__(self, experimentation=Physics(2000,50,1,1), constraint = 'cube',\
                 nb_blocks=20, noise = 0.05,\
                 folder = './', im_set="Set1",batch_size=[50,5],\
                 lr=1e-3, nb_epochs=[10,1]):
        """
        Parameters
        ----------
           
        """
        super(FBRestNet, self).__init__()   
        # physical information
        self.physics    = experimentation
        self.noise      = noise
        self.constr     = constraint
        # training information
        self.lr_i       = lr
        self.nb_epochs  = nb_epochs[0]
        self.freq_val   = nb_epochs[1]
        self.nb_blocks  = nb_blocks
        self.nsamples   = batch_size[0]
        self.train_size = batch_size[1] # training set 
        self.val_size   = 1            # and validation set/test set 
        self.im_set     = im_set
        self.loss_fn    = torch.nn.MSELoss(reduction='mean')
        # saving info
        self.path       = folder
        # requires regularisation
        self.regul      = (noise>0)&(self.physics.m>20)
        # model creation
        self.model      = MyModel(self.physics,noisy=self.regul,nL=self.nb_blocks,constr=self.constr)
#========================================================================================================
#========================================================================================================
    def LoadParam(self):
        """
        Load the parameters of a trained model (in Trainings)
        """
        path_model = self.path+'Trainings/param_{}_{}_'.format(\
                    self.physics.a,self.physics.p)+self.constr+'.pt'
        self.model.load_state_dict(torch.load(path_model));
        self.model.eval() # be sure to run this step!
#========================================================================================================
#========================================================================================================    
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
        # Test_cuda()
        # self.device   =
        # self.dtype    =
        # Recuperation des donnees
        nx             = self.physics.nx
        m              = self.physics.m
        a              = self.physics.a
        noise          = self.noise
        nsample        = self.nsamples
        im_set         = self.im_set
        Teig           = np.diag(self.physics.eigm**(-a))
        Pelt           = self.physics.Operators()[3]
        # Initialisation
        color          = ('b','g','r')
        #
        liste_l_trsf   = []
        liste_tT_trsf  = []
        #
        save_lisse     = []
        save_l_trsf    = []
        save_blurred   = []
        save_blurred_n = []
        save_tT_trsf   = []
        # Upload Data
        # path : './MyResNet/Datasets/Images/'
        for folder, subfolders, filenames in os.walk(self.path+'Datasets/Images/'+im_set+'/'): 
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
                    yp[:ncrop]    = yp[ncrop-1]
                    yp[nx-ncrop:] = 0
                    yp[yp<0]      = 0
                    yp            = yp/np.amax(yp)
                    # filtering high frequencies
                    fmax          = 4*m//5
                    filtre        = Physics(nx,fmax)
                    yp            = filtre.BasisChange(yp)
                    x_true        = filtre.BasisChangeInv(yp)
                    x_true[x_true<0] = 0
                    if self.constr == 'cube':
                        x_true += 0.01
                        x_true  = 0.9*x_true/np.amax(x_true)
                    if self.constr == 'slab':
                        u      = 1/nx**2*np.linspace(1,nx,nx)
                        x_true = 0.5*x_true/np.dot(u,x_true)
                    # reshaping in channelxm
                    x_true  = x_true.reshape(1,-1)
                    # save
                    save_lisse.append(x_true.squeeze())
                    # Etape 2 : passage dans la base de T^*T
                    yp          = self.physics.BasisChange(x_true)
                    x_true_trsf = yp.reshape(1,-1)
                    # save
                    liste_l_trsf.append(x_true_trsf)
                    save_l_trsf.append( x_true_trsf.squeeze())
                    #  Etape 3 : obtenir les images bruitees par l' operateur d' ordre a
                    # transform
                    x_blurred  = self.physics.Compute(x_true).squeeze()
                    # save
                    save_blurred.append(x_blurred)
                    # Etape 4 : noise 
                    vn          = np.zeros(m)
                    vn          = np.random.randn(m)*self.physics.eigm**(-a)
                    vn_elt      = self.physics.BasisChangeInv(vn)
                    vn_elt      = vn_elt/np.linalg.norm(vn_elt)
                    x_blurred_n = x_blurred + self.noise*np.linalg.norm(x_blurred)*vn_elt
                    # save
                    save_blurred_n.append(x_blurred_n)
                    # Etape 5 : bias
                    x_b       = self.physics.ComputeAdjoint(x_blurred.reshape(1,-1))
                    vn[:fmax] = np.zeros(fmax)
                    x_b      += (Teig.dot(vn)).reshape(1,-1) # noise
                    x_b       = x_b.reshape(1,-1)
                    # and save
                    liste_tT_trsf.append(x_b)
                    save_tT_trsf.append(x_b.squeeze())
        # Export data in .csv
        if save =='yes':
            seq = 'a{}_'.format(self.physics.a) + self.constr 
            # initial signal, no noise, elt basis
            np.savetxt(self.path+'Datasets/Signals/data_l_'+seq+'.csv',      save_lisse,   delimiter=', ', fmt='%12.8f')
            # initial signal, no noise, eig basis
            np.savetxt(self.path+'Datasets/Signals/data_l_trsf_'+seq+'.csv', save_l_trsf,  delimiter=', ', fmt='%12.8f')
            # blurred signal, no noise, elt basis
            np.savetxt(self.path+'Datasets/Signals/data_b_'+seq+'.csv',    save_blurred, delimiter=', ', fmt='%12.8f')
            # blurred signal, noisy, elt basis
            np.savetxt(self.path+'Datasets/Signals/data_bn_'+seq+'_n{}'.format(noise)+'.csv',  save_blurred_n, delimiter=', ', fmt='%12.8f')
            # Transposed blurred signal, noisy, eig basis
            np.savetxt(self.path+'Datasets/Signals/data_tTb_'+seq+'_n{}'.format(noise)+'.csv',  save_tT_trsf, delimiter=', ', fmt='%12.8f')
        # Tensor completion
        x_tensor = torch.FloatTensor(liste_l_trsf) # signal in cos/eig basis
        y_tensor = torch.FloatTensor(liste_tT_trsf)# blurred and noisy signal in elt basis
        #
        dataset = TensorDataset(y_tensor[:nsample], x_tensor[:nsample])
        l       = len(dataset)
        ratio   = 2*l//3
        train_dataset, val_dataset = random_split(dataset, [ratio, l-ratio])
        #
        train_loader = DataLoader(train_dataset, batch_size=self.train_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)
        #
        return train_loader, val_loader
#========================================================================================================
#========================================================================================================
    def LoadDataSet(self):
        """
        Dreates the appropriate loader for the training and validation sets
        when the dataset is already created.
        """
        #
        nsample = self.nsamples
        #
        seq = 'a{}_'.format(self.physics.a) + self.constr
        dfl     = pd.read_csv(self.path+'Datasets/Signals/data_l_trsf_'+seq+'.csv', sep=',',header=None)
        dfb    = pd.read_csv(self.path+'Datasets/Signals/data_tTb_'+seq+'_n{}'.format(self.noise)+'.csv', sep=',',header=None)
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
#========================================================================================================
#========================================================================================================    
    def train(self,train_set,val_set,test_lipschitz=True,save_model=False):
        """
        Trains iRestNet.
        """      
        # to store results
        nb_epochs  = self.nb_epochs
        nb_val     = self.nb_epochs//self.freq_val
        loss_train =  np.zeros(nb_epochs)
        loss_val   =  np.zeros(nb_val)
        lip_cste   =  np.zeros(nb_val)
        # defines the optimizer
        lr_i       = self.lr_i
        optimizer  = torch.optim.Adam(filter(lambda p: p.requires_grad,self.model.parameters()),lr=self.lr_i)   
        # filtering parameter
        fmax     = self.physics.m//3  
        # trains for several epochs
        for epoch in range(0,self.nb_epochs): 
            # sets training mode
            self.model.train()
            # modifies learning rate
            if epoch>0:
                lr_i      = self.lr_i*0.9 
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.model.parameters()), lr=lr_i)
            # TRAINING
            # goes through all minibatches
            for i,minibatch in enumerate(train_set):
                [y, x]    = minibatch    # get the minibatch
                x_bias    = Variable(y,requires_grad=False)
                x_true    = Variable(x,requires_grad=False) 
                # definition of the initialisation tensor
                # x_init   = torch.zeros(x_bias.size())
#                 inv      = np.diag(self.physics.eigm**(2*self.physics.a))
#                 tTTinv   = MyMatmul(inv)
#                 x_init[:fmax]   = tTTinv(y)[:fmax] # no filtration of high frequences
                x_init   = Variable(y,requires_grad=False)
                # prediction
                x_pred    = self.model(x_init,x_bias) 
                # Computes and prints loss
                loss               = self.loss_fn(x_pred, x_true)
                norm               = torch.norm(x_true.detach())
                loss_train[epoch] += torch.Tensor.item(loss/norm)
                # 
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
                    loss_init = 0
                    for i,minibatch in enumerate(val_set):
                        [y, x] = minibatch            # gets the minibatch
                        x_true  = Variable(x,requires_grad=False)
                        x_bias  = Variable(y,requires_grad=False)
                        # definition of the initialisation tensor
                        x_init   = Variable(y,requires_grad=False)
                        # prediction
                        x_pred  = self.model(x_init,x_bias).detach()
                        # computes loss on validation set
                        norm    = torch.norm(x_true.detach())
                        loss    = self.loss_fn(x_pred, x_true)
                        loss_val[epoch//self.freq_val] += torch.Tensor.item(loss/norm)
                        loss_init += torch.Tensor.item(self.loss_fn(x_init, x_true)/torch.norm(x_true.detach()))
                    # normalisation
                    loss_val[epoch//self.freq_val] = loss_val[epoch//self.freq_val]/i
                # print stat
                print("epoch : ", epoch," ----- ","validation : ",loss_val[epoch//self.freq_val])
                print("initial error :",loss_init/i)
                # Test Lipschitz
                lip_cste[epoch//self.freq_val] = self.model.Lipschitz()
                
            
        #=======================
        # training is finished
        print('-----------------------------------------------------------------')
        print('Training is done.')
        print('-----------------------------------------------------------------')
        
        # Plots
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(np.linspace(0,1,nb_epochs),loss_train,label = 'train')
        ax1.plot(np.linspace(0,1,nb_val),loss_val,label = 'val')
        ax1.legend()
        ax2.plot(np.linspace(0,1,nb_val),lip_cste,'r-')
        ax2.set_title("Lipschitz constant")
        plt.show()
        #
        print("Final Lipschitz constant = ",lip_cste[-1])
        # Export lip curve
        Export_Data(np.linspace(0,nb_val-1,nb_val),\
                    lip_cste,
                    self.path+'Redaction/data',\
                    'lip{}_{}_{}_{}.pt'.format(\
                    self.physics.nx,self.physics.m,self.physics.a,self.physics.p))
        # Save model
        if save_model:
            torch.save(self.model.state_dict(), self.path+'Trainings/param_{}_{}_'.format(\
            self.physics.a,self.physics.p)+self.constr+'.pt')
#========================================================================================================
#========================================================================================================    
    def test(self,data_set):    
        """
        Parameters
        ----------
            dataset        (Dataloader): the test set
        """
        # initial
        torch_zeros = Variable(torch.zeros(1,1,self.physics.m),requires_grad=False)
        counter     = 0
        avrg        = 0
        # filtering parameter
        fmax     = 4*self.physics.m//5
        # gies through the minibatch
        with torch.no_grad():
            self.model.eval()
            loss_init = 0
            for i,minibatch in enumerate(data_set):
                [y, x] = minibatch            # gets the minibatch
                x_true = Variable(x,requires_grad=False)
                x_bias = Variable(y,requires_grad=False)
                # definition of the initialisation tensor
                x_init   = torch.zeros(x_bias.size())
                inv      = np.diag(self.physics.eigm**(2*self.physics.a))
                tTTinv   = MyMatmul(inv)
                x_init[:,:,:fmax] = tTTinv(y)[:,:,:fmax] # filtration of high frequences
                x_init   = Variable(x_init,requires_grad=False)
                # prediction
                x_pred    = self.model(x_init,x_bias)
                # compute loss
                loss   = torch.Tensor.item(self.loss_fn(x_pred, x_true))
                norm   = torch.Tensor.item(self.loss_fn(torch_zeros, x_true))
                loss_init += torch.Tensor.item(self.loss_fn(x_init, x_true)/torch.norm(x_true.detach()))
                # l_loss.append(loss/norm)
                avrg    += loss/norm
                counter += 1
        # Plots
        xtc = x_true.numpy()[0,0]
        xpc = x_pred.numpy()[0,0]
        xic = x_init.numpy()[0,0]
        xt  = self.physics.BasisChangeInv(xtc)
        xp  = self.physics.BasisChangeInv(xpc)
        xi  = self.physics.BasisChangeInv(xic)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(xtc,'+',label = 'true')
        ax1.plot(xpc,'kx',label = 'pred')
        ax1.plot(xic,label = 'init')
        ax1.legend()
        ax2.plot(np.linspace(0,1,self.physics.nx),xt,'+',label = 'true')
        ax2.plot(np.linspace(0,1,self.physics.nx),xp,label = 'pred')
        ax2.set_title("Comparaison")
        ax2.legend()
        plt.show()
        #
        print("Erreur relative : ",avrg/counter)
        print("Erreur initiale : ",loss_init/counter)
        # return 
        return avrg/counter
#========================================================================================================
#======================================================================================================== 
    def test_gauss(self, noise = 0.05):
        # Gaussienne 
        nx    = self.physics.nx
        m     = self.physics.m
        t     = np.linspace(0,1,nx)
        gauss = np.exp(-(t-0.5)**2/(0.1)**(2))
        # filtering high frequencies
        fmax          = m//2
        filtre        = Physics(nx,fmax)
        gauss         = filtre.BasisChange(gauss)
        gauss         = filtre.BasisChangeInv(gauss)
        gauss[gauss<0] = 0
        #
        if self.constr == 'cube':
            gauss = 0.9*gauss/np.amax(gauss)+0.01
        if self.constr == 'slab':
            u      = 1/nx**2*np.linspace(1,nx,nx)
            gauss = 0.5*gauss/np.dot(u,gauss)
        # export
        Export_Data(t,gauss,'./Redaction/data','gauss_'+self.constr)
        # obtenir les images bruitees par l' operateur d' ordre a
        # transform
        x_blurred  = self.physics.Compute(gauss).squeeze()
        yp         = self.physics.BasisChange(x_blurred)
        # Etape 4 : noise 
        vn          = np.zeros(m)
        vn[fmax:]   = np.random.randn(m-fmax)
        vn          = noise*np.linalg.norm(yp)*vn/np.linalg.norm(vn)
        x_blurred_n = x_blurred + self.physics.BasisChangeInv(vn)
        # Etape 5 : bias
        x_b  = self.physics.ComputeAdjoint(x_blurred_n)
        # passage float tensor
        x_bias    = Variable(torch.FloatTensor(x_b.reshape(1,1,-1)),requires_grad=False)
        # definition of the initialisation tensor
        with torch.no_grad():
        # tests on validation set
            self.model.eval()
            x_init   = torch.zeros(x_bias.shape)
            tTTinv   = MyMatmul(np.diag(self.physics.eigm**(2*self.physics.a)))
            x_init[:,:,fmax//2] = tTTinv(x_bias)[:,:,fmax//2]  # filtration of high frequences
            x_init   = Variable(x_init.reshape(1,1,-1),requires_grad=False)
            # prediction
            x_pred   = self.model(x_init,x_bias)
            xpc      = x_pred.detach().numpy()[0,0,:]
            xp       = self.physics.BasisChangeInv(xpc)
            xp[xp<0] = 0
        # export
        print(type(self.constr))
        Export_Data(t,xp,'./Redaction/data',\
                        'gauss_pred_a{}'.format(self.physics.a)+self.constr)
        # plot
        plt.plot(t,gauss)
        plt.plot(t,xp)
        print("x-xp/x =",np.linalg.norm(xp-gauss)/np.linalg.norm(gauss))
        
