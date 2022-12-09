import os
from typing import Tuple

import mlflow
from mlflow import log_artifact, log_metric, log_metrics, log_params
from torch import save
from torch.nn.functional import embedding

# from libs.dataloader import *
from .hyparam_searcher import choose_value_of_c
from .module import GSlayer, PencilModel, PredNet, RejNet
from .pencil_evaluator import test
from .pencil_trainer import train
from .utils import *
from copy import deepcopy

class Pencil():
    def __init__(self, mode, select_genes=True, seed=1234, data_name='test', expr_id='test', model_types=None, dropouts=None, mlflow_record=False):
        '''Do not include 'simul'" in your data_name.'''
        
        # setup_seed(seed) #fix the random state.
        self.seed = seed
        self.mlflow_record = mlflow_record
        # print('mlflow_record:', self.mlflow_record)
        if self.mlflow_record:
            try:
                mlflow.create_experiment(data_name)
            except Exception as e:
                print(e)
        
            try:
                mlflow.set_experiment(data_name)
            except Exception as e:
                print(e)
            
        try:
            # os.makedirs('./results/%s' % data_name)
            os.makedirs('./results/%s/model' % data_name)
        except:
            pass
        
        try:
            # os.makedirs('./results/%s' % data_name)
            os.makedirs('./results/%s/py/%s' % (data_name, expr_id))
        except:
            pass
        
        self.mode = mode
        self.select_genes = select_genes
        self.data_name = data_name
        self.expr_id = expr_id
        # self.model = None
        
        if self.mode == 'regression':
            self.loss_type, self.rej_type = ('sml1', 'NGLR')
        elif self.mode == 'multi-classification':
            self.loss_type, self.rej_type = ('ce', 'Sigmoid')
        # else:
        #     loss_type, rej_type = ('neg-product', 'LR')
        
        if dropouts is None:
            if self.mode == 'regression':
                self.dropouts = [0.2, 0.0]
            else:
                self.dropouts = [0.0, 0.0]
        else:
            self.dropouts = dropouts
        self.model_types = model_types
    
    def set_model(self, data, labels, hide_feats_pred=[200], hide_feats_rej=[200]):
        in_feat = data.shape[1]
        
        if self.model_types is None:
            if self.mode == 'multi-classification':
                model_types = ['linear', 'non-linear']
            elif self.mode == 'regression':
                model_types = ['non-linear', 'non-linear']
        else:
            model_types = self.model_types
            
        predictor = PredNet(in_features=in_feat, out_features=get_out_dim(labels, self.mode), hide_features=hide_feats_pred, model_type=model_types[0], dropout=self.dropouts[0], mlflow_record=self.mlflow_record)
        
        rejector = RejNet(in_features=in_feat, hide_features=hide_feats_rej, model_type=model_types[1], tanh=True, dropout=self.dropouts[1], mlflow_record=self.mlflow_record)
        
        if self.select_genes:
            gslayer =  GSlayer(in_features=in_feat)
        else:
            gslayer = None
            
        self.model = PencilModel(predictor, rejector, gslayer, mlflow_record=self.mlflow_record)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
                
    def fit(self, data, labels, c=None, lambda_L1=1e-5, lambda_L2=1e-3, lr=0.01, epochs=500, pre_train_epochs=None, class_weights=None, loss_type=None, rej_type=None, laplacian_reg=False, shuffle_rate=1/3, range_of_c=[0.0, 2.0], use_cuda=True, plot_show=False, batch_size=None, once_load_to_gpu=True, savefig=True, **kwargs):
        setup_seed(self.seed) #fix the random state.
        
        log_file = open('./results/%s/py/%s/report.txt' % (self.data_name, self.expr_id), 'w')
        if loss_type is None:
            loss_type = self.loss_type
        if rej_type is None:
            rej_type = self.rej_type
        
        print('dataset: %s, expr_id: %s' % (self.data_name, self.expr_id))
        print('scheme: %s, %s' % (loss_type, rej_type))
        
        lambda_laplacian = 1e-4
        
        cmax = range_of_c[1]
        cmin = range_of_c[0]
        
        if pre_train_epochs is None:
            if self.mode == 'regression':
                pre_train_epochs = 0
            elif self.mode == 'multi-classification':
                pre_train_epochs = 500
            
        self.set_model(data, labels)
        self.model.train()
        
        # mlflow.start_run()
        if self.mlflow_record:
            log_params({
                'expr_id': self.expr_id,
                'mode': self.mode,
                "shuffle_rate": shuffle_rate,
                'laplacian_reg': laplacian_reg
            })
        
        if laplacian_reg:
            _, laplacian = sparse_graph(data, k=5)
        else:
            laplacian = None
        
        Xtr, Ytr = data, labels
        
        if c is None:
            print('searching c...')
            c = choose_value_of_c(
                self.model, Xtr, Ytr, 
                cmax=cmax,
                cmin=cmin, 
                bisect_eps=0.02, 
                shuffle_rate=shuffle_rate, 
                check_thr=0.1,
                dataset_name=self.data_name, 
                expr_id=self.expr_id, 
                mode=self.mode,
                loss_type=loss_type, 
                rej_type=rej_type, 
                class_weights=class_weights,
                epochs=epochs,
                lr=lr, 
                laplacian=laplacian,
                lambda_laplacian=lambda_laplacian, 
                use_cuda=use_cuda,
                pre_train_epochs=pre_train_epochs,
                lambda_L1=lambda_L1, 
                lambda_L2=lambda_L2,
                savefig=savefig,
                once_load_to_gpu=once_load_to_gpu,
                mlflow_record=self.mlflow_record
            )
            print('searched c:', c)
        else:
            print('c:', c)
            
        ####################### train ######################################################  
        # setup_seed(self.seed) #fix the random state. #TODO: check if fix again? 
        # print('locate here')
        self.model = train(
            self.model, Xtr, Ytr,
            mode=self.mode,
            c=c,
            loss_type=loss_type, 
            rej_type=rej_type, 
            class_weights=class_weights,
            epochs=epochs, 
            lr=lr, 
            use_lr_scheme=True, 
            use_cuda=use_cuda, 
            dataset_name=self.data_name, 
            expr_id=self.expr_id, 
            plot_loss=True,
            laplacian=laplacian,
            lambda_laplacian=lambda_laplacian,
            log_file=log_file,
            pre_train_epochs=pre_train_epochs,
            lambda_L1=lambda_L1, 
            lambda_L2=lambda_L2,
            plot_show=plot_show,
            batch_size=batch_size,
            savefig=savefig,
            once_load_to_gpu=once_load_to_gpu,
            mlflow_record=self.mlflow_record
        )
        
        log_file.close()
        plt.close()
        
    def transform(self, data, labels=None, class_names=None, anno_file=None, emd=None, embedding_file=None, group_info_file=None, use_cuda=True, plot_show=True, embedding_name=None, savefig=True, res_to_csv=True, **kwargs):
        self.model.eval()
        
        if labels is None:
            Xte = torch.Tensor(data)
            if use_cuda:
                if torch.cuda.is_available():
                    # print('cuda is available.')
                    Xte = Xte.cuda()
                    # Yte = Yte.cuda()
                    self.model.cuda()
                else:
                    print('cuda is not available, and cpu is used.')
            else:
                self.model.cpu()

            with torch.no_grad():
                h, r = self.model(Xte)
                h = h.detach().cpu()
                r = r.detach().cpu().view(-1).numpy()
        else:
            log_file = open('./results/%s/py/%s/report.txt' % (self.data_name, self.expr_id), 'w')
            
            Xte, Yte = data, labels
            
            if emd is None:
                if embedding_file is None:
                    try:
                        emd = pd.read_csv('./datasets/%s/%s.csv' % (self.data_name, embedding_name), index_col=0).values
                    except:
                        emd = None
                else:
                    emd = pd.read_csv(embedding_file, index_col=0).values
    
            try:
                # groups_info = pd.read_csv('./datasets/%s/groups_info.csv' % self.data_name, sep=',', index_col=0)
                groups_info = pd.read_csv(group_info_file, sep=',', index_col=0)
            except:
                groups_info = None
            
            h, r = test[self.mode](
                self.model,
                Xte, Yte,
                use_cuda=use_cuda, 
                dataset_name=self.data_name, 
                expr_id=self.expr_id, 
                class_names=class_names, 
                anno_file = anno_file,
                umap_plot=plot_show, 
                umap_embedding=emd,
                groups_info_to_check=groups_info,
                log_file=log_file,
                plot_show=plot_show,
                savefig=savefig,
                res_to_csv=res_to_csv,
                mlflow_record=self.mlflow_record
            )
            log_file.close()
            if self.mlflow_record:
                log_artifact('./results/%s/py/%s/report.txt' % (self.data_name, self.expr_id))
            plt.close()

        if self.mode != 'regression':
            return torch.argmax(h, 1).numpy(), r
        else:
            return h.numpy(), r
    
    def fit_transform(self, data, labels, test=False, savefig=True, **kwargs):
        self.fit(data, labels, savefig=savefig, **kwargs)
        if test:
            h, r = self.transform(data, labels, savefig=savefig, **kwargs)
        else:
            h, r = self.transform(data)
        
        return h, r
    
    def gene_weights(self, plot=False, savefig=True, save_path=None):
        if self.select_genes:
            w = self.model.gslayer.select_weight.detach().cpu().numpy()
            np.savetxt('./results/%s/py/%s/selected_weight.csv' % (self.data_name, self.expr_id), w, delimiter=',')
            if self.mlflow_record:
                log_artifact('./results/%s/py/%s/selected_weight.csv' % (self.data_name, self.expr_id))
            if plot:
                plt.figure(figsize=(35, 7))
                plt.bar(np.arange(w.shape[0]), w)
                if savefig:
                    if save_path is None:  
                        plt.savefig('./results/%s/py/%s/selected_weight.pdf' % (self.data_name, self.expr_id))
                        plt.show()
                        if self.mlflow_record:
                            log_artifact('./results/%s/py/%s/selected_weight.pdf' % (self.data_name, self.expr_id)) 
                    else:
                        plt.save(save_path)
                        plt.show()
                        if self.mlflow_record:
                            log_artifact(save_path)
            return w
        else:
            print('select_genes is False.')
        
        
            
