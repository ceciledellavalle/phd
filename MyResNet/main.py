#global import
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys
#local import
from MyResNet.myfunc import Physics
from MyResNet.myfunc import MyMatmul
from MyResNet.model import MyModel

        
class MyRestNet_class(nn.Module):
    """
    Includes the main training and testing methods of iRestNet.
    Attributes
    ----------
        im_size        (numpy array): image size
        path_test              (str): path to the folder containing the test sets
        path_train             (str): path to the training set folder 
    """
    def __init__(self, condition, folders, mode='test', 
                 lr_i=[1e-2,5], nb_epochs=[40,40,600], nb_blocks=20, 
                 batch_size=[10,10,1], loss_type='MSE'):
        """
        Parameters
        ----------
            folders            (list): list of str, paths to the folder containing (i) the test sets, (ii) the training, (iii) saved models  
            lr                 (list): list of two elements, first one is the initial learning rate to train the layers, 
                                       second one is the number of epochs after which the learning rate is multiplied by 0.9 (default is [5e-3,5])
            nb_epochs          (list): list of three integers, number of epochs for training the first layer, the remaining layers, 
                                       and the last 10 layers + lpp, respectively (default is [40,40,600])      
            nb_blocks           (int): number of unfolded iterations (default is 40)    
            batch_size         (list): list of three integers, number of images per batch for training, validation and testing, respectively (default is [10,10,1])                
            loss_type           (str): name of the training loss (default is 'OT')  
        """
        super(iRestNet_class, self).__init__()   
        # physical data
        self.a  = condition[0]   
        self.p  = condition[1]   
        self.nx = condition[2]
        self.tensor_list = Physics(a,p,nx)#to compute 
        self.Tt          = MyMatmul(self.tensor_list[1].T)
        #
        self.mass = 1
        self.U    = troch.FloatTensor(np.ones(self.nx))
        # unpack information about saving folder
        self.path_test, self.path_train, self.path_save = folders
        # training information
        self.lr_i       = lr
        self.nb_epochs  = nb_epochs  # nb of epochs for the first layers, other layers trained in a greedy fashion, last layers+lpp
        self.nb_blocks  = nb_blocks
        self.batch_size = batch_size # training set and validation set/test set 
        self.loss_type  = loss_type  # 'OT' or 'MSE'
        #
        if self.mode!='test':
            #definition of the loss function 
            if self.loss_type=='OT':
                self.loss_fun = Wass_loss() 
            elif self.loss_type=='MSE':
                self.loss_fun = torch.nn.MSELoss(size_average=True)
        #
        self.model = MyModel(self.tensor_list,self.mass,self.U,self.nb_blocks)

    
    def CreateLoader(self):
        """
        According to the mode, creates the appropriate loader for the training and validation sets.
        """
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size[0], shuffle=True)
        self.val_loader   = DataLoader(val_data, batch_size=self.batch_size[1], shuffle=False)
        self.size_train   = len([n for n in os.listdir(os.path.join(self.path_train,'train'))])
        self.size_val     = len([n for n in os.listdir(os.path.join(self.path_train,'val'))])
        
    def train(self):
        """
        Trains iRestNet.
        Parameters
        ----------
            block (int): number of the layer to be trained, numbering starts at 0 (default is 0)
        """      
        # trains the first layer
        # to store results
        loss_epochs  =  np.zeros(self.nb_epochs[0])
        loss_train   =  np.zeros((2,2,self.nb_epochs[0]))
        loss_val     =  np.zeros((2,2,self.nb_epochs[0]))
        loss_min_val =  float('Inf')
        self.CreateLoader()
        # defines the optimizer
        lr_i        = self.lr_i
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.parameters()),lr=lr_i)

        #==========================================================================================================
        # trains for several epochs
        for epoch in range(0,self.nb_epochs[0]): 
            # sets training mode
            self.model.train()
            # modifies learning rate
             if epoch>0:
                lr_i        = lr_i*0.9 
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.parameters()), lr=lr_i)
            # goes through all minibatches
            for i,minibatch in enumerate(self.train_loader,0):
                [x_true, x_blurred] = minibatch    # get the minibatch
                x_true    = Variable(x_true.type(self.dtype),requires_grad=False)
                x_init    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                # ATTENTION : on ne calcule pas le gradient en fonction de la deuxième sortie (à réfléchir)
                Ttx_init  = Tt(x_init).detach()     # do not compute gradient
                x_pred    = self.model(x_init,Ttx_init,self.mode) 
                    
                # Computes and prints loss
                loss                = self.loss_fun(x_pred, x_true)
                loss_epochs[epoch] += torch.Tensor.item(loss)
                sys.stdout.write('\r(%d, %3d) minibatch loss: %5.4f '%(epoch,i,torch.Tensor.item(loss)))
                    
                # sets the gradients to zero, performs a backward pass, and updates the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                 # for statistics
                loss_train[:,:,epoch] += compute_stat(x_true, x_blurred, x_pred, self.size_train)

            # saves images and model state  
             if epoch%20==0:
                ### SAVE stat
                # torch.save(self.model.state_dict(),os.path.join(folder,'trained_model.pt'))
                pass

            # tests on validation set
            self.model.eval()      # evaluation mode
            loss_current_val = 0
            for minibatch in self.val_loader:
                [x_true, x_blurred] = minibatch            # gets the minibatch
                x_true    = Variable(x_true.type(self.dtype),requires_grad=False)
                x_blurred = Variable(x_blurred.type(self.dtype),requires_grad=False)
                Ttx_init  = self.Tt(x_blurred).detach()        # does not compute gradient
                x_pred    = self.model(x_blurred,Ttx_init,self.mode) 
                    
                # computes loss on validation set
                loss_current_val += torch.Tensor.item(self.loss_fun(x_pred, x_true))

            # prints statistics
            self.PrintStatistics(epoch, loss_epochs[epoch],lr)
            self.SaveLoss(epoch, loss_epochs, self.mode)
        #==========================================================================================================
        # training is finished
        print('-----------------------------------------------------------------')
        print('Training is done.')
        self.SaveLoss(epoch,loss_epochs,self.mode)
        print('-----------------------------------------------------------------')

    
    def test(self, dataset, save_gamma_mu_lambda='no'):    
        """
        Parameters
        ----------
            dataset        (str): name of the test set
            save_gamma_mu_lambda: indicates if the user wants to save the values of the estimated hyperparameters (default is 'no')
        """
        # for RGB and gray images        
        path_savetest                   = os.path.join(self.path_save,'Results_on_Testsets',dataset)
        path_dataset                    = os.path.join(self.path_test, self.name_kernel+self.name_std, dataset)
        if save_gamma_mu_lambda=='no':
            print('Saving restaured images in %s ...'%(path_savetest),flush=True)
            # creates directory for saving results
            if not os.path.exists(path_savetest):
                os.makedirs(path_savetest)
            data          = MyTestset(folder=path_dataset)
            loader        = DataLoader(data, batch_size=self.batch_size[2], shuffle=False)
            # evaluation mode
            self.model.eval() 
            self.last_layer.eval()
            for minibatch in tqdm(loader,file=sys.stdout):
                [im_names, [yes_no, x_blurred]] = minibatch # gets the minibatch
                x_blurred     = Variable(x_blurred.type(self.dtype), requires_grad=False)
                Ht_x_blurred  = self.Ht(x_blurred)       
                x_pred        = self.model(x_blurred,Ht_x_blurred.detach(),self.mode)
                x_pred        = x_pred.detach()
                x_pred        = self.sigmoid(x_pred + self.last_layer(x_pred))
                # saves restored images
                for j in range(len(im_names)):
                    sio.savemat(os.path.join(path_savetest,im_names[j]),{'image':RGBToGray(
                        yes_no[j],x_pred.data[j]).permute(1,2,0).cpu().numpy().astype('float64')})
        else:
            print('Saving restaured images in %s ...'%(path_savetest),flush=True)
            print('Saving stepsize, barrier parameter and regularization parameter in %s ...'%(save_gamma_mu_lambda),flush=True)
            data          = MyTestset(folder=path_dataset)
            loader        = DataLoader(data, batch_size=1, shuffle=False)
            # evaluation mode
            self.model.eval() 
            self.last_layer.eval()
            for minibatch in tqdm(loader,file=sys.stdout):
                [im_names, [yes_no, x_blurred]] = minibatch # gets the minibatch
                path_gamma_mu_lambda = os.path.join(save_gamma_mu_lambda,im_names[0][0:-4])
                if not os.path.exists(path_gamma_mu_lambda):
                    os.makedirs(path_gamma_mu_lambda)
                x_blurred     = Variable(x_blurred.type(self.dtype), requires_grad=False)
                Ht_x_blurred  = self.Ht(x_blurred)       
                x_pred        = self.model(x_blurred,Ht_x_blurred.detach(),self.mode,save_gamma_mu_lambda=path_gamma_mu_lambda)
                
        
    def PrintStatistics(self, train, val, epoch, loss, lr):
        """
        Prints information about the training.
        Parameters
        ----------
        train (list): size 2*2, average PSNR and SSIM on the training set and on the deblurred training images
        val   (list): size 2*2, average PSNR and SSIM on the validation set and on the deblurred validation images
        epoch  (int): epoch number
        loss (float): value of the training loss function
        lr   (float): learning rate
        """
        print('-----------------------------------------------------------------')
        print('[%d]'%(epoch),'average', self.loss_type,': %5.5f'%(loss), 'lr %.2E'%(lr))
        print('     Training set:') 
        print('         PSNR blurred = %2.3f, PSNR pred = %2.3f'%(train[0,0],train[0,1]))
        print('         SSIM blurred = %2.3f, SSIM pred = %2.3f'%(train[1,0],train[1,1]))
        print('     Validation set:') 
        print('         PSNR blurred = %2.3f, PSNR pred = %2.3f'%(val[0,0],val[0,1]))
        print('         SSIM blurred = %2.3f, SSIM pred = %2.3f'%(val[1,0],val[1,1]))
    
 
