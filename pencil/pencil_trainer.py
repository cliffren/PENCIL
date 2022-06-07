import math
import time

import matplotlib.pyplot as plt
from sympy import beta
import torch
import torch.nn.functional as F
import torch.optim as optim
from mlflow import log_artifact, log_metric, log_metrics, log_params

from .loss_function import binary_loss, rejection_loss, pseudo_mse_loss
from torch.utils.data import DataLoader, TensorDataset
# from libs.utils import *

def cyclical_lr(step_size, min_lr=3e-4, max_lr=3e-3):
    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, step_size)

    # Additional function to see where on the cycle we are
    def relative(it, step_size):
        cycle = math.floor(1 + it / (2 * step_size))
        x = abs(it / step_size - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda

def train_clf_only(pencil, Xtr, Ytr, 
                   mode=None, 
                   loss_type='hinge', 
                   epochs=100, 
                   batch_size=None,
                   lr=0.01, 
                   use_lr_scheme=True, 
                   class_weights=[1.0, 2.0], 
                   use_cuda=True, 
                   once_load_to_gpu=True,
                   dataset_name='simulated', 
                   expr_id='', 
                   plot_loss=True, 
                   plot_show=False, 
                   silence=False,
                   lambda_L1=1e-6,
                   savefig=True
                   ):
    '''
    train the model without rejection.

    Parameters
    ----------
    pencil: torch.nn.Module.
        The pencil model icluding predictor, rejector and gslayer(gene-select layer).
    Xtr: numpy.ndarray.
        The training data.
    Ytr: numpy.ndarray.
        The traing labels.
    loss_type: str. 
        The type of classification-loss function. See `binary_loss_func`.
        hinge | soft-hinge | neg-product | ce.
        hinge: hinge loss. $max(0, 1- y_true*y_predict)$.
        soft-hinge: a simple smooth approximation of hinge loss.
        neg-product: $max(0, 1- y_true*y_predict)$.
        ce: cross entropy loss.
    class_weights: list of floats or None. 
        The item is the weight for each class. If it is None, the weight is set to 1 for each class.
        For regression, each sample is regarded as one class, so class_weights must be a list with length equaling number of samples.
    '''
    Xtr = torch.Tensor(Xtr)
    if mode=='multi-classification':
        Ytr = torch.LongTensor(Ytr)
    else:
        Ytr = torch.Tensor(Ytr)

    sample_weights = torch.ones(Ytr.shape[0])
    if class_weights is not None:
        if mode=='binary-classification':
            sample_weights[Ytr==1] = class_weights[0]
            sample_weights[Ytr==-1] = class_weights[1]
        elif mode=='multi-classification':
            class_weights_ = torch.Tensor(class_weights)
            sample_weights = class_weights_[Ytr]
        else:
            sample_weights = torch.Tensor(class_weights)
    
    cuda_used = False
    if use_cuda:
        if torch.cuda.is_available():
            cuda_used = True
            if not silence:
                print('cuda is available.')
            if once_load_to_gpu:
                Xtr = Xtr.cuda()
                Ytr = Ytr.cuda()
                sample_weights = sample_weights.cuda()
                
            pencil.cuda()
            # zero = zero.cuda()
        else:
            if not silence:
                print('cuda is not available, and cpu is used.') 

    optimizer = optim.Adam(pencil.parameters(), lr=lr, weight_decay=1e-4)
    clr = cyclical_lr(step_size=100, min_lr=1e-4, max_lr=1.0) 
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
    
    if batch_size is None:
        dataloader_tr = [[Xtr, Ytr, sample_weights]]
    else:
        dataset_tr = TensorDataset(Xtr, Ytr, sample_weights)
        dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True)
        
    L=[]
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader_tr):
            X, Y, sample_weights_batch = batch
            if cuda_used and (not once_load_to_gpu):
                X = X.cuda()
                Y = Y.cuda()
                sample_weights_batch = sample_weights_batch.cuda()
                
            if (pencil.gslayer is not None) and (lambda_L1 > 0):
                L1_reg = torch.sum(torch.abs(pencil.gslayer.select_weight))
            else:
                L1_reg = 0
                
            # Forward pass: Compute predicted y by passing x to the model
            h, _ = pencil(X)
            h = torch.squeeze(h, 1)

            if mode=='binary-classification':
                e = binary_loss(y_true=Y, y_pred=h, mtype=loss_type)
            elif mode=='multi-classification':
                assert loss_type=='ce', 'only ce loss can work for multiclass classificaiton.'
                e = F.cross_entropy(h, Y, reduction='none')
                e = 1 - torch.exp(-e) # map the ce-loss to [0, 1], that is 1-p.
            elif mode=='regression':
                if loss_type=='mse':
                    e = F.mse_loss(h, Y, reduction='none')
                elif loss_type=='sml1':    
                    beta = 1.0
                    e = F.smooth_l1_loss(h, Y, reduction='none', beta=beta) * beta
                elif loss_type=='pmse':
                    e = pseudo_mse_loss(h, Y, reduction='none')

            loss = e * sample_weights_batch
            norm_factor = 1 / sample_weights_batch.sum()
            loss = torch.sum(loss) * norm_factor + L1_reg * lambda_L1

            if torch.isnan(loss):
                if not silence:
                    log_metric('real_epochs', epoch + 1)
                    print('Terminate training since loss equals NAN.')
                break

            L.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if use_lr_scheme:
                scheduler.step()

        if not silence:    
            if epoch % 20 == 0:
                print('epoch=%d, loss=%.4f' % (epoch, loss))
                # print('lr:', scheduler.get_lr())
                
    data_id = round(Xtr[0,].sum().item(), 3)
    torch.save(pencil.state_dict(),'./results/%s/model/pretrain_%s_%s.pth' %   (dataset_name, expr_id, str([mode, loss_type, epochs, lr, data_id])))
    if not silence:        
        if plot_loss:
            plt.plot(L)
            plt.title('Loss')
            plt.grid()
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            if savefig:
                plt.savefig('./results/%s/py/%s/loss_pretrain.pdf' % (dataset_name, expr_id))
            if plot_show:
                plt.show()

        pencil.load_state_dict(torch.load('./results/%s/model/pretrain_%s_%s.pth' %   (dataset_name, expr_id, str([mode, loss_type, epochs, lr, data_id])), map_location='cpu'))
        
    return pencil

def train(pencil, Xtr, Ytr, 
          mode='multi-classification', 
          c=0.4, 
          loss_type='hinge', rej_type='NGLR', 
          batch_size=None,
          epochs=100, lr=0.01, use_lr_scheme=True, 
          class_weights=None, 
          use_cuda=True, 
          once_load_to_gpu=True,
          dataset_name='simulated', expr_id='', 
          plot_loss=True, 
          plot_show=False, 
          laplacian=None, 
          lambda_laplacian=1e-4, 
          log_file=None, 
          silence=False, 
          pre_train_epochs=500, 
          lambda_L1=1e-4, 
          lambda_L2=1e-3,
          load_pretrained_model=False,
          savefig=True):
    '''
    train the model with rejection.

    Parameters
    ----------
    pencil: torch.nn.Module.
        The pencil model icluding predictor, rejector and gslayer(gene-select layer).
        
    Xtr: numpy.ndarray.
        The training data.
    Ytr: numpy.ndarray.
        The traing labels.
    mode: str.
        The learning purpose. multi-classification | regression. (binary-classification can also be performed through multi-classification.)
    c: double or sequence which length equals Xtr.shape[0].
        The rejection cost for each sample. 
    loss_type: str. 
        The type of classification-loss function. See `binary_loss_func`.
        hinge | soft-hinge | neg-product | ce.
        hinge: hinge loss. $max(0, 1- y_true*y_predict)$.
        soft-hinge: a simple smooth approximation of hinge loss.
        neg-product: $max(0, 1- y_true*y_predict)$.
        ce: cross entropy loss.
    rej_type: str. 
        The type of rejcttion-loss.  
        LR | GLR | NGLR | Sigmoid.
        LR: from "learning with rejecition". 
        GLR: for "Generalized learning with rejection".
        NGLR: new GLR.
    class_weights: list of floats or None. 
        The item is the weight for each class. If it is None, the weight is set to 1 for each class. 
        It can also be a weight list of length equals to the number of classes for each sample-cell.
        For regression, each sample is regarded as one class, so class_weights must be a list with length equaling number of samples.
    laplacian: numpy.ndarray.
        The laplacian matrix for laplacian-regularization of r(x).
    lambda_laplacian: float. The default is 1e-4.
        The multiplier for the laplacian-regularization. $\lambda r(x)^T L r(x)$.
    silence: bool.
        If true, all of printed content will not be displayed and mlflow will not work.
    pre_train: bool.
        If true, the model mlp_class will be pre-trained using `train_clf_only`.
    log_file: None or file object.
        To log some information.

    Returns
    -------
    Trained mlp_class. 
    Trained mlp_rej.

    See Also
    --------
    `train_clf_only`

    '''
    start_time = time.time()
    if laplacian is None:
        lambda_laplacian = None
        
    if pencil.gslayer is None:
        lambda_L1 = 0
        
    if not silence:
        if class_weights is None:
            class_weights_record = 'None'
        elif len(class_weights) == Xtr.shape[0]:
            class_weights_record = 'sample-weights'
        else:
            class_weights_record = str(class_weights)
            
        log_params({
            'c': c,
            'loss_type': loss_type,
            'rej_type': rej_type,
            'class_weights': class_weights_record,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'use_lr_scheme': use_lr_scheme,
            'pretrain_epochs': pre_train_epochs,
            'lambda_laplacian': lambda_laplacian,
            'lambda_L1': lambda_L1, 
            'lambda_L2': lambda_L2
        })

    # pretrain
    if pre_train_epochs > 0:
        if load_pretrained_model:
            try:
                data_id = round(Xtr[0,].sum(), 3)
                path = './results/%s/model/pretrain_%s_%s.pth' %   (dataset_name, expr_id, str([mode, loss_type, pre_train_epochs, lr, data_id]))
                pencil.load_state_dict(torch.load(path, map_location='cpu'))
                if not silence:
                    print('load pretrained model from %s' % path)
            except:
                if not silence:
                    print('pretrain %d epochs...' % pre_train_epochs)
                pencil = train_clf_only(pencil, Xtr, Ytr, mode=mode, loss_type=loss_type, batch_size=batch_size, epochs=pre_train_epochs, lr=lr, use_lr_scheme=True, class_weights=class_weights, use_cuda=True, silence=True, dataset_name=dataset_name, expr_id=expr_id)
        else:
            if not silence:
                    print('pretrain %d epochs...' % pre_train_epochs)
            pencil = train_clf_only(pencil, Xtr, Ytr, mode=mode, loss_type=loss_type, batch_size=batch_size, epochs=pre_train_epochs, lr=lr, use_lr_scheme=True, class_weights=class_weights, use_cuda=True, silence=True, dataset_name=dataset_name, expr_id=expr_id, once_load_to_gpu=once_load_to_gpu)
            
    Xtr = torch.Tensor(Xtr)
    if mode=='multi-classification':
        Ytr = torch.LongTensor(Ytr)
    else:
        Ytr = torch.Tensor(Ytr)
        
    sample_weights = torch.ones(Ytr.shape[0])
    if class_weights is not None:
        if mode=='binary-classification':
            sample_weights[Ytr==1] = class_weights[0]
            sample_weights[Ytr==-1] = class_weights[1]
        elif mode=='multi-classification':
            if len(class_weights) == Xtr.shape[0]:
                sample_weights = torch.Tensor(class_weights)
            else:
                class_weights_ = torch.Tensor(class_weights)
                sample_weights = class_weights_[Ytr]
        else:
            sample_weights = torch.Tensor(class_weights)
    
    cuda_used = False
    if use_cuda:
        if torch.cuda.is_available():
            cuda_used = True
            if not silence:
                print('cuda is available.')
            if once_load_to_gpu:
                Xtr = Xtr.cuda()
                Ytr = Ytr.cuda()
                sample_weights = sample_weights.cuda()
                
            pencil.cuda()
            try:
                c = torch.Tensor(c)
                c = c.cuda()
            except:
                pass

            if laplacian is not None:
                assert batch_size is None, "Not support for batch-training"
                laplacian = torch.Tensor(laplacian)
                laplacian = laplacian.cuda()
        else:
           print('cuda is not available, and cpu is used.') 

    optimizer_predictor = optim.Adam(pencil.predictor.parameters(), lr=lr, weight_decay=lambda_L2)
    optimizer_rejector = optim.Adam(pencil.rejector.parameters(), lr=lr, weight_decay=1e-3)
    if pencil.gslayer is not None:
        optimizer_gslayer =  optim.Adam(pencil.gslayer.parameters(), lr=lr)

    clr = cyclical_lr(step_size=100, min_lr=1e-4, max_lr=1.0)
    # check max_lr: 1.0 or lr.
    scheduler_predictor = torch.optim.lr_scheduler.LambdaLR(optimizer_predictor, [clr])
    scheduler_rejector = torch.optim.lr_scheduler.LambdaLR(optimizer_predictor, [clr])
    if pencil.gslayer is not None:
        scheduler_gslayer = torch.optim.lr_scheduler.LambdaLR(optimizer_predictor, [clr])
    
    
    if laplacian is not None:
        assert batch_size is None, 'laplacian regularizaion do not support batch train.'
    if batch_size is None:
        dataloader_tr = [[Xtr, Ytr, sample_weights]]
    else:
        dataset_tr = TensorDataset(Xtr, Ytr, sample_weights)
        dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True, num_workers=0)
        
    L=[]
    stop_early = False
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader_tr):
            X, Y, sample_weights_batch = batch
            if cuda_used and (not once_load_to_gpu):
                X = X.cuda()
                Y = Y.cuda()
                sample_weights_batch = sample_weights_batch.cuda()
                
            if pencil.gslayer is not None:
                L1_reg = torch.sum(torch.abs(pencil.gslayer.select_weight))
            else:
                L1_reg = 0
                
            h, r = pencil(X)
            h = torch.squeeze(h, 1)
            r = torch.squeeze(r, 1)
            
            # print(r.shape)
            if mode=='binary-classification':
                e = binary_loss(y_true=Y, y_pred=h, mtype=loss_type)
            elif mode=='multi-classification':
                assert loss_type=='ce', 'only ce loss can work for multiclass classificaiton.'
                e = F.cross_entropy(h, Y, reduction='none')
                e = 1 - torch.exp(-e) #map ce-loss to [0, 1], that is 1-p.
            elif mode=='regression':
                # assert loss_type=='mse', 'only mse loss can work for regression classificaiton.'
                if loss_type=='mse':
                    e = F.mse_loss(h, Y, reduction='none')
                elif loss_type=='sml1':    
                    beta = 1.0
                    e = F.smooth_l1_loss(h, Y, reduction='none', beta=beta)
                    # e = F.mse_loss(h, Ytr, reduction='none')
                elif loss_type=='pmse':
                    e = pseudo_mse_loss(h, Y, reduction='none')

            l1 = rejection_loss(e, r, c, mtype=rej_type)
            loss_r = l1 * sample_weights_batch    
            norm_factor = 1 / sample_weights_batch.sum()

            if laplacian is not None:
                loss_r = torch.sum(loss_r) * norm_factor + lambda_laplacian * torch.mm(torch.mm(r.view(1, -1), laplacian), r.view(-1, 1)) + L1_reg * lambda_L1
            else:
                loss_r = torch.sum(loss_r) * norm_factor + L1_reg * lambda_L1

            if torch.isnan(loss_r):
                if not silence:
                    log_metric('real_epochs', epoch + 1)
                stop_early = True 
                print('Terminate training since loss equals NAN.')
                break

            L.append(loss_r.item())

            optimizer_predictor.zero_grad()
            optimizer_rejector.zero_grad()
            if pencil.gslayer is not None:
                optimizer_gslayer.zero_grad()
            loss_r.backward()

            optimizer_predictor.step()
            optimizer_rejector.step()
            if pencil.gslayer is not None:
                optimizer_gslayer.step()
            
            mean_e, mean_r = e.mean(), r.mean()

            if not silence:
                log_metrics({
                    'loss': loss_r.item(),
                    'mean_e': mean_e.item(),
                    'mean_r': mean_r.item()
                }, step=epoch)
            
            # print('epoch=%d, batch=%d, loss=%.4f, mean_e=%.4f, mean_r=%.4f, L1_reg=%.4f' % (epoch, i, loss_r, mean_e, mean_r, L1_reg))   
             
        if use_lr_scheme:
            scheduler_predictor.step()
            scheduler_rejector.step()
            if pencil.gslayer is not None:
                scheduler_gslayer.step()
            # print(scheduler_predictor.get_last_lr())
                
        if not silence:
            if epoch % 20 == 0:
                print('epoch=%d, loss=%.4f, mean_e=%.4f, mean_r=%.4f, L1_reg=%.4f' % (epoch, loss_r, mean_e, mean_r, L1_reg))
                # print('lr:', scheduler.get_lr())
        
    if not silence:
        torch.save(pencil.state_dict(),'./results/%s/model/pencil_%s.pth' %   (dataset_name, expr_id))
        
    if (not stop_early) and (not silence):
        log_metric('real_epochs', epochs)
            
    if plot_loss:
        plt.plot(L)
        plt.title('Loss')
        plt.grid()
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        if savefig:
            plt.savefig('./results/%s/py/%s/loss.pdf' % (dataset_name, expr_id))
            log_artifact('./results/%s/py/%s/loss.pdf' % (dataset_name, expr_id))
        
        if plot_show:
            plt.show()
            
    if not silence:
        pencil.load_state_dict(torch.load('./results/%s/model/pencil_%s.pth' %   (dataset_name, expr_id), map_location='cpu'))
        
        print("---train time: %s seconds ---\n" % (time.time() - start_time))
        log_file.write("---train time: %s seconds ---\n" % (time.time() - start_time))

    plt.close()
    return pencil

