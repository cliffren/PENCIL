import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score as auc_roc

import pandas as pd
from sklearn.metrics import classification_report
from copy import deepcopy
import time 
from mlflow import log_metrics, log_artifact
from .utils import *

def binary_class_test(pencil, Xte, Yte, use_cuda=True, dataset_name='simulated', expr_id='', class_names=None, anno_file=None, plot_show=False, umap_plot=False, umap_embedding=None, groups_info_to_check=None, log_file=None, savefig=True, res_to_csv=True):
    '''
    Test function for binary-classification.

    Parameters
    ----------
    pencil: torch.nn.Module.
        The pencil model icluding predictor, rejector and gslayer(gene-select layer).
    Xte: numpy.ndarray.
        The testing data.
    Yte: numpy.ndarray.
        The testing labels.
    anno_file: str.
        The path to the file including the infomation of cell_ids and labels.
    umap_plot: bool.
        If true, plot the umap of results.
    umap_embeding: None or numpy.ndarray.
        If None, use umap_reducer to caculate the embedding. See `plot_umap`.
    groups_info_to_check: pandas.DataFrame.
        The infomation to check the results by counting the abundance of each group in each category. See `abundance_of_groups`.
    log_file: None or file object.
        To log some information.
        
    Returns
    -------
    h: numpy.ndarray.
        The prediction results of testing data. $h(x)$. 
    r: numpy.ndarray.
        The rejection results of testing data. $r(x)$.    
    '''
    pencil.eval()
    
    start_time = time.time()
    Xte = torch.Tensor(Xte)
    # Ytr = torch.LongTensor(labels)
    # Yte = torch.Tensor(Yte)

    cuda_used = False
    if use_cuda:
        if torch.cuda.is_available():
            cuda_used = True
            # print('cuda is available.')
            Xte = Xte.cuda()
            pencil.cuda()
        else:
           print('cuda is not available, and cpu is used.')
    else:
        pencil.cpu()
    
    with torch.no_grad():
        h, r = pencil(Xte)
        h = h.detach().cpu().numpy().flatten()
        r = r.detach().cpu().numpy().flatten()

    f = log_file
    try:
        auc_c = auc_roc(Yte, h)
        auc_r = auc_roc(Yte[r>0], h[r>0])

        plot_roc([Yte, Yte[r>0]] , [h, h[r>0]],  ['without rejction', 'with rejction'])
        if savefig:
            plt.savefig('./results/%s/py/%s/auc.pdf' % (dataset_name, expr_id))
            log_artifact('./results/%s/py/%s/auc.pdf' % (dataset_name, expr_id)) 
        if plot_show:
            plt.show()
        

        print("AUC without rejection=", auc_c)
        print("AUC with rejection=", auc_r)
        
        f.write("AUC without rejection=%f\n" % auc_c)
        f.write("AUC with rejection=%f\n" % auc_r)
        
        log_metrics({
            'auc': auc_c,
            'auc_rej': auc_r,
            # 'auc_rej_over_rate_rej': auc_r / len(r[r<0]) * len(r)
        })

    except Exception as e:
        print(e)

    print ("Number of examples rejected=", len(r[r<0]), "/", len(r))
    f.write("Number of examples rejected=%d/%d.\n" % (len(r[r<0]), len(r)))

    true_label_of_rej = Yte[r<0]
    n_pos_rej = (true_label_of_rej == 1).sum()
    n_neg_rej = (true_label_of_rej==-1).sum()

    print('Reject %d positive samples and %d negtive samples.\n' % (n_pos_rej, n_neg_rej))
    f.write('Reject %d positive samples and %d negtive samples.\n' % (n_pos_rej, n_neg_rej))

    # print ("auc_rej/rate_rej=", auc_r / len(r[r<0]) * len(r))
    # f.write("auc_rej/rate_rej=%f\n" % (auc_r / len(r[r<0]) * len(r)))
    
    log_metrics({
            'num_rejected': np.sum(r<0),
            'num_pos_rejected': n_pos_rej,
            'num_neg_rejected': n_neg_rej,
            'num_samples': len(r)
        })

    if 'simul' in dataset_name:
        Xte = Xte.cpu().numpy()
        # output of classification network
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plotit(Xte, Yte, clf=pencil.predictor, transform = None, conts =[-1,0,1], markers = ('.','.'), use_cuda=cuda_used)
        plt.title('h(x)')
        # output of rejection network
        plt.subplot(122)
        plotit(Xte,Yte,clf=pencil.rejector, transform = None, conts =[0], ccolors = ['g'], markers = ('.','.') , use_cuda=cuda_used)
        plt.title('r(x)')
        if savefig:
            plt.savefig('./results/%s/py/%s/test_result.pdf' % (dataset_name, expr_id))
            log_artifact('./results/%s/py/%s/test_result.pdf' % (dataset_name, expr_id)) 
        if plot_show:
            plt.show()
        
    else:  
        result_info = np.vstack((Yte, h, r))
        result_info = result_info.T

        save_file = './results/%s/py/%s/high_dimens_result.csv' % (dataset_name, expr_id)
        np.savetxt(save_file, result_info, delimiter=',')
        log_artifact(save_file) 

        # pre_labels_path = './results/%s/predicted_labels.csv' % (dataset_name)
        if res_to_csv:
            res_df = res_to_labels(result_info, anno_file, class_names=class_names, keep_origin_label_for_rest=False)
            # res_df.to_csv(pre_labels_path, index=False) #save to one file for R-script.
            # log_artifact(pre_labels_path)

            pre_labels_path_ = './results/%s/py/%s/predicted_labels.csv' % (dataset_name, expr_id)
            res_df.to_csv(pre_labels_path_, index=False) #save to one file with the expr_id tag.

        print('--- without rejection ---')
        f.write('--- without rejection ---' + '\n')

        y_class_cluster = deepcopy(h)
        y_class_cluster[y_class_cluster>=0] = 1
        y_class_cluster[y_class_cluster<0] = -1
        tmp_clfr = classification_report(Yte, y_class_cluster)
        print(tmp_clfr)
        f.write(tmp_clfr)
        f.write('\n')

        print('--- with rejection ---')
        f.write('--- with rejection ---' + '\n')
        y_class_remain = deepcopy(h[r>0])
        y_class_remain[y_class_remain>=0] = 1
        y_class_remain[y_class_remain<0] = -1
        
        tmp_clfr = classification_report(Yte[r>0], y_class_remain)
        print(tmp_clfr)
        f.write(tmp_clfr)
        f.write('\n')

        if groups_info_to_check is not None:
            overlap_g1_in_pos, overlap_g2_in_pos, overlap_g1_in_neg, overlap_g3_in_neg, out_str = abundance_of_groups(groups_info_to_check, y_class_remain, r)
            log_metrics({
                'overlap_g1_in_pos' : overlap_g1_in_pos,
                'overlap_g2_in_pos' : overlap_g2_in_pos,
                'overlap_g1_in_neg': overlap_g1_in_neg,
                'overlap_g3_in_neg': overlap_g3_in_neg,
            })
            f.write(out_str)
            print(out_str)

        if umap_plot:
            plot_umap(Xte.cpu().numpy(), Yte, y_class_cluster, r, embedding=umap_embedding)
            if savefig:
                plt.savefig('./results/%s/py/%s/test_result.pdf' % (dataset_name, expr_id))
                log_artifact('./results/%s/py/%s/test_result.pdf' % (dataset_name, expr_id)) 
            if plot_show:
                plt.show()
            

    print("---test time: %s seconds ---" % (time.time() - start_time))
    f.write("---test time: %s seconds ---\n" % (time.time() - start_time))

    return h, r

def mul_class_test(pencil, Xte, Yte, use_cuda=True, dataset_name='simulated', expr_id='', class_names=None, anno_file=None, umap_plot=True, plot_show=False, umap_embedding=None, log_file=None, savefig=True, res_to_csv=True, **kwargs):
    '''
    Test function for multi-classification.

    See Also
    --------
    `binary_class_test`
    
    '''
    pencil.eval()
    
    start_time = time.time()
    Xte = torch.Tensor(Xte)
    # Ytr = torch.LongTensor(labels)
    # Yte = torch.Tensor(Yte)

    cuda_used = False
    if use_cuda:
        if torch.cuda.is_available():
            cuda_used = True
            # print('cuda is available.')
            Xte = Xte.cuda()
            # Yte = Yte.cuda()
            pencil.cuda()
        else:
           print('cuda is not available, and cpu is used.')
    else:
        pencil.cpu()

    with torch.no_grad():
        h, r = pencil(Xte)
        h = h.detach().cpu()
        r = r.detach().cpu().view(-1).numpy()

    f = log_file
    print ("Number of examples rejected=", len(r[r<0]), "/", len(r))
    f.write("Number of examples rejected=%d/%d.\n" % (len(r[r<0]), len(r)))

    true_label_of_rej = Yte[r<0]
    true_label_of_rej = np.array(class_names)[true_label_of_rej]
    true_label_of_rej = pd.DataFrame(true_label_of_rej, columns=['num_of_rejcted'])
    rej_info = true_label_of_rej.value_counts()
    print(rej_info)
    f.write(str(rej_info))
    f.write('\n')

    log_metrics({
            'num_rejected': len(r[r<0]),
            # 'num_rejected_each': rej_info.values.tolist(),
            'num_samples': len(r)
        })
    
    pred_labels = torch.argmax(h, 1).numpy()
    result_info = np.vstack((Yte, pred_labels, r))
    result_info = result_info.T

    save_file = './results/%s/py/%s/high_dimens_result.csv' % (dataset_name, expr_id)
    np.savetxt(save_file, result_info, delimiter=',')
    log_artifact(save_file) 

    if 'simul' not in dataset_name and res_to_csv:
        # pre_labels_path = './results/%s/predicted_labels.csv' % (dataset_name)
        res_df = res_to_labels(result_info, anno_file, class_names=class_names, keep_origin_label_for_rest=False)
        # res_df.to_csv(pre_labels_path, index=False) #save to one file for R-script.
        # log_artifact(pre_labels_path)

        pre_labels_path_ = './results/%s/py/%s/predicted_labels.csv' % (dataset_name, expr_id)
        res_df.to_csv(pre_labels_path_, index=False) #save to one file with the expr_id tag.
        log_artifact(pre_labels_path_)

    print('--- without rejection ---')
    f.write('--- without rejection ---' + '\n')
    tmp_clfr = classification_report(Yte, pred_labels, labels=np.arange(len(class_names)), target_names=class_names)
    print(tmp_clfr)
    f.write(tmp_clfr)
    f.write('\n')

    print('--- with rejection ---')
    tmp_clfr = classification_report(Yte[r>0], pred_labels[r>0], labels=np.arange(len(class_names)), target_names=class_names)
    print(tmp_clfr)
    f.write(tmp_clfr)
    f.write('\n')

    if umap_plot:
        plot_mul_class_umap(Yte, pred_labels, r, embedding=umap_embedding, X=Xte.cpu().numpy(), class_names=class_names)
        if savefig:
            plt.savefig('./results/%s/py/%s/test_result.pdf' % (dataset_name, expr_id))
            log_artifact('./results/%s/py/%s/test_result.pdf' % (dataset_name, expr_id))
        if plot_show:
            plt.show()
        
    
    print("---test time: %s seconds ---" % (time.time() - start_time))
    f.write("---test time: %s seconds ---\n" % (time.time() - start_time))

    return h, r


def reg_test(pencil, Xte, Yte, use_cuda=True, dataset_name='simulated', expr_id='', class_names=None, anno_file=None, umap_plot=True, plot_show=False, umap_embedding=None, log_file=None, savefig=True, **kwargs):
    '''
    Test function for regression.

    See Also
    --------
    `binary_class_test`
    
    '''
    pencil.eval()
    
    start_time = time.time()
    Xte = torch.Tensor(Xte)
    # Ytr = torch.LongTensor(labels)
    # Yte = torch.Tensor(Yte)

    cuda_used = False
    if use_cuda:
        if torch.cuda.is_available():
            cuda_used = True
            # print('cuda is available.')
            Xte = Xte.cuda()
            # Yte = Yte.cuda()
            pencil.cuda()
        else:
           print('cuda is not available, and cpu is used.')
    else:
        pencil.cpu()
    
    with torch.no_grad():
        h, r = pencil(Xte)
        h = h.detach().cpu()
        r = r.detach().cpu().view(-1).numpy()

    # print(Yte.shape, h.shape, r.shape)
    result_info = np.vstack((Yte, h.reshape(-1), r))
    result_info = result_info.T

    save_file = './results/%s/py/%s/high_dimens_result.csv' % (dataset_name, expr_id)
    np.savetxt(save_file, result_info, delimiter=',')
    log_artifact(save_file) 

    f = log_file
    print ("Number of examples rejected=", len(r[r<0]), "/", len(r))
    f.write("Number of examples rejected=%d/%d.\n" % (len(r[r<0]), len(r)))

    if umap_plot:
        plot_reg_umap(Yte, h, r, embedding=umap_embedding, X=Xte.cpu().numpy())
        if savefig:
            plt.savefig('./results/%s/py/%s/test_result.pdf' % (dataset_name, expr_id))
            log_artifact('./results/%s/py/%s/test_result.pdf' % (dataset_name, expr_id))
        if plot_show:
            plt.show()
        

    print("---test time: %s seconds ---" % (time.time() - start_time))
    f.write("---test time: %s seconds ---\n" % (time.time() - start_time))

    return h, r
  
test = {
    'binary-classification': binary_class_test,
    'multi-classification': mul_class_test,
    'regression': reg_test
}