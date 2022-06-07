import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
from sklearn.metrics import roc_curve, auc
import itertools
import warnings
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import pdist, squareform

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def get_out_dim(labels, mode):
    out = len(list(set(labels)))
    if mode == 'regression' or mode == 'binary-classification':
        return 1
    else:
        return out

def get_class_names(filepath):
    with open(filepath) as f:
        lines = f.readlines()[0][:-1]
        # print(lines)
        class_names = lines.split(',')
    return class_names

def sparse_graph(data, k):
    '''get the graph and laplacian matrix.'''
    # calculate directly. 
    # TODO: load from the seurat object through file.  

    graph_mat = squareform(pdist(data))
    sparse_graph_mat = np.zeros(graph_mat.shape)

    for i in range(graph_mat.shape[0]):
        tmp = graph_mat[i]
        sparse_graph_mat[i, np.argsort(tmp)[0:k]] = 1
    
    sparse_graph_mat += sparse_graph_mat.T
    sparse_graph_mat[sparse_graph_mat == 2] = 1
    laplacian = np.diag(graph_mat.sum(1)) - graph_mat
    
    return sparse_graph_mat, laplacian

def abundance_of_groups(groups_df, y_class_remain, y_r):
    '''Count the abundance of each group in each category.'''
    groups_remain = groups_df.values[y_r>0]

    overlap_g1_in_pos = (groups_remain[y_class_remain == 1] == 'Group1').sum() / (y_class_remain == 1).sum()
    overlap_g2_in_pos = (groups_remain[y_class_remain == 1] == 'Group2').sum() / (y_class_remain == 1).sum()

    overlap_g1_in_neg = (groups_remain[y_class_remain == -1] == 'Group1').sum() / (y_class_remain == -1).sum()
    overlap_g3_in_neg = (groups_remain[y_class_remain == -1] == 'Group3').sum() / (y_class_remain == -1).sum()
    
    out_str = '--- abundance of groups ---\n'
    out_str += 'overlap_g1_in_pos: %f' % overlap_g1_in_pos + '\n' + \
        'overlap_g2_in_pos: %f' % overlap_g2_in_pos + '\n' + \
        'overlap_g3_in_pos: %f' % (1 - overlap_g2_in_pos - overlap_g1_in_pos) + '\n\n' + \
        'overlap_g1_in_neg: %f' % overlap_g1_in_neg + '\n' + \
        'overlap_g3_in_neg: %f' % overlap_g3_in_neg + '\n' + \
        'overlap_g2_in_neg: %f' % (1 - overlap_g3_in_neg -overlap_g1_in_neg) + '\n'
        
    
    return overlap_g1_in_pos, overlap_g2_in_pos, overlap_g1_in_neg, overlap_g3_in_neg, out_str

def res_to_labels(results, anno_file=None, class_names=None, keep_origin_label_for_rest=False):
    '''Map the prediction results to the class-names.
    
    Parameters
    ----------
    results: numpy.ndarray.
        A matrix with 3 columns corresponding to (Y_true, h, r).
    anno_file: str.
        The path to the file including the infomation of cell_ids and labels.
    keep_origin_label_for_rest: bool. False by default.
        If true, keeping the origin labels instead of the predicted labels for the samples that were not rejected.

    '''
    
    if anno_file is not None:
        label_df = pd.read_csv(anno_file, sep=',')
    else:
        data = np.hstack((np.arange(results.shape[0]).reshape(-1,1), results[:,0].reshape(-1,1)))
        label_df = pd.DataFrame(data=data, columns=['cell_id', 'label'])

    Yt, h, r = results[:, 0], results[:, 1], results[:, 2]
    pred_label = np.zeros(h.shape, dtype=object)

    if len(class_names)==2 and h.min() < 0:
        if keep_origin_label_for_rest:
            pred_label[Yt>0] = class_names[0]
            pred_label[Yt<0] = class_names[1]
        else:
            pred_label[h>0] = class_names[0]
            pred_label[h<=0] = class_names[1]
        pred_label[r<=0] = 'Rejected'
    else:
        pred_label[r<=0] = 'Rejected'
        class_names_ = np.array(class_names)
        if keep_origin_label_for_rest:
            pred_label[r>0] = class_names_[Yt[r>0]]
        else:
            h = np.array(h, dtype=np.int)
            pred_label[r>0] = class_names_[h[r>0]]

    Yt = np.array(Yt, dtype=np.int)
    unique_Yt = list(set(Yt.tolist()))

    if len(unique_Yt) < len(class_names):
        bool_reserved_ids = (label_df.values[:, 1] == class_names[unique_Yt[0]])
        for i in range(1, len(unique_Yt)):
            bool_reserved_ids = bool_reserved_ids | (label_df.values[:, 1] == class_names[unique_Yt[i]])
        data = np.hstack((label_df.values[bool_reserved_ids, 0].reshape(-1, 1), pred_label.reshape(-1, 1)))
    else:
        data = np.hstack((label_df.values[:, 0].reshape(-1, 1), pred_label.reshape(-1, 1)))

    df = pd.DataFrame(data=data, columns=['cellid', 'predicted_label'])

    return df

def plot_roc(label_list, pred_list, legend_label_list):
    '''plot roc curves for a list of prediction results.'''

    plt.figure(figsize=(6,6))
    for i in range(len(label_list)):
        label, pred, legend_label = label_list[i], pred_list[i], legend_label_list[i]

        fpr, tpr, threshold = roc_curve(label, pred)
        roc_auc = auc(fpr, tpr)

        # plt.title('Validation ROC')
        plt.plot(fpr, tpr, label = '%s(%0.3f, %d samples)' % (legend_label, roc_auc, len(label)))

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc = 'lower right')

def plot_umap(X, Ytr, Y_pred, Y_r, embedding=None):
    
    if embedding is None:
        import umap 
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(X)

    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    # print(embedding.shape)

    Y = Ytr.reshape(-1,1)
    data = np.hstack((embedding[:, 0:2], Y))
    df = pd.DataFrame(data=data, columns=['umap1', 'umap2', 'real_class'])
    sns.scatterplot(x='umap1', y='umap2', data=df, hue='real_class', ax=axs[0], palette="Set1", hue_order=[-1, 1], markers=['.'], s=10)

    Y_ = np.array(Y_pred.reshape(-1,1), dtype=np.object)
    Y_[Y_r<=0] = 'rej'
    data = np.hstack((embedding[:, 0:2], Y_))
    df = pd.DataFrame(data=data, columns=['umap1', 'umap2', 'pred_class'])
    sns.scatterplot(x='umap1', y='umap2', data=df, hue='pred_class', ax=axs[1], palette="Set1", hue_order=[-1, 1, 'rej'], markers=['.'], s=10)

    rthr = 0.0
    if embedding is None:
        embedding_ = reducer.fit_transform(X[Y_r > rthr])
    else:
        embedding_ = embedding[Y_r > rthr]

    Ytr_ = Ytr.reshape(-1,1)[Y_r > rthr]
    data = np.hstack((embedding_[:, 0:2], Ytr_))
    df = pd.DataFrame(data=data, columns=['umap1', 'umap2', 'real_class'])
    sns.scatterplot(x='umap1', y='umap2', data=df, hue='real_class', ax=axs[2], palette="Set1", hue_order=[-1, 1], markers=['.'], s=10)
    # plt.show()

def plotit(X, Y=None,clf=None,  conts = None, ccolors = ('b','k','r'), colors = ('c','y'), markers = ('s','o'), hold = False, transform = None, use_cuda=False, **kwargs):
    """
    @author: Dr. Fayyaz Minhas
    @author-email: afsar at pieas dot edu dot pk
    2D Scatter Plotter for Classification
    A function for showing data scatter plot and classification boundary
    of a classifier for 2D data
        X: nxd  matrix of data points
        Y: (optional) n vector of class labels
        clf: (optional) classification/discriminant function handle
        conts: (optional) contours (if None, contours are drawn for each class boundary)
        ccolors: (optional) colors for contours   
        colors: (optional) colors for each class (sorted wrt class id)
            can be 'scaled' or 'random' or a list/tuple of color ids
        markers: (optional) markers for each class (sorted wrt class id)
        hold: Whether to hold the plot or not for overlay (default: False).
        transform: (optional) a function handle for transforming data before passing to clf
        kwargs: any keyword arguments to be passed to clf (if any)        
    """
    if clf is not None and X.shape[1]!=2:
        warnings.warn("Data Dimensionality is not 2. Unable to plot.")
        return
    if markers is None:
        markers = ('.',)

    d0,d1 = (0,1)
    minx, maxx = np.min(X[:,d0]), np.max(X[:,d0])
    miny, maxy = np.min(X[:,d1]), np.max(X[:,d1])
    eps=1e-6

    if Y is not None:
        classes = sorted(set(Y))
        if conts is None:
            conts = list(classes)        
        vmin,vmax = classes[0]-eps,classes[-1]+eps
    else:
        vmin,vmax=-2-eps,2+eps
        if conts is None:            
            conts = sorted([-1+eps,0,1-eps])
    if clf is not None:
        npts = 150
        x = np.linspace(minx,maxx,npts)
        y = np.linspace(miny,maxy,npts)
        t = np.array(list(itertools.product(x,y)))
        if transform is not None:
            t = transform(t)
        t=Variable(torch.from_numpy(t)).type(torch.FloatTensor)
        if use_cuda:
            t = t.cuda()
        with torch.no_grad():
            z = clf(t,**kwargs)
        
        z = np.reshape(z.cpu().numpy(),(npts,npts)).T        
        extent = [minx,maxx,miny,maxy]
        
        plt.contour(x, y, z, conts, linewidths=[2], colors=ccolors, extent=extent)
        #plt.imshow(np.flipud(z), extent = extent, cmap=plt.cm.Purples, vmin = -2, vmax = +2); plt.colorbar()
        plt.pcolormesh(x, y, z, cmap=plt.cm.Purples, vmin=vmin, vmax=vmax, shading='auto')
        plt.colorbar()
        plt.axis([minx, maxx, miny, maxy])
    
    if Y is not None:        
        for i,y in enumerate(classes):
            if colors is None or colors=='scaled':
                cc = np.array([[i,i,i]])/float(len(classes))
            elif colors =='random':
                cc = np.array([[np.random.rand(),np.random.rand(),np.random.rand()]])
            else:
                cc = colors[i%len(colors)]
            mm = markers[i%len(markers)]
            plt.scatter(X[Y==y,d0],X[Y==y,d1], marker = mm,c = cc, s = 30)     
         
    else:
        plt.scatter(X[:,d0],X[:,d1],marker = markers[0], c = 'k', s = 5)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')   

    if not hold:
        plt.grid()        
        # plt.show()

def plot_mul_class_umap(Y, Y_pred, y_r, embedding=None, X=None, size=10, class_names=None):

    class_names = np.array(class_names)
    if embedding is None:
        import umap 
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(X)

    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    data = np.hstack((embedding, Y.reshape(-1,1)))
    n_classes = len(list(set(Y.tolist())))

    colors = sns.color_palette('Set1')
    palette=colors[0:n_classes]
    hue_order = np.sort(list(set(Y.tolist()))).tolist()
    hue_order = class_names[hue_order].tolist()

    df = pd.DataFrame(data=data, columns=['embd1', 'embd2', 'real_class'])

    tmp = df.loc[:, 'real_class'].values
    tmp = np.array(tmp, dtype=np.int)
    df.loc[:, 'real_class'] = class_names[tmp]

    sns.scatterplot(x='embd1', y='embd2', data=df, hue='real_class', ax=axs[0], palette=palette, hue_order=hue_order, s=size)

    Yp = np.array(Y_pred.reshape(-1,1), dtype=np.object)
    Yp[y_r<0] = 'rejected'
    data = np.hstack((embedding, Yp))
    df = pd.DataFrame(data=data, columns=['embd1', 'embd2', 'pred_class'])

    tmp = df.loc[y_r>0, 'pred_class'].values
    tmp = np.array(tmp, dtype=np.int)
    df.loc[y_r>0, 'pred_class'] = class_names[tmp]

    hue_order.insert(0, 'rejected')   
    palette.insert(0, colors[-1])

    sns.scatterplot(x='embd1', y='embd2', data=df, hue='pred_class', ax=axs[1], palette=palette, hue_order=hue_order, s=size)

    # embedding = reducer.fit_transform(X[y_r>0])
    # data = np.hstack((embedding, Yp[y_r>0]))
    # df = pd.DataFrame(data=data, columns=['embd1', 'embd2', 'pred_class'])
    # sns.scatterplot(x='embd1', y='embd2', data=df, hue='pred_class', ax=axs[1], palette=palette[:-1], hue_order=hue_order[:-1], s=size)
    # plt.show()


def plot_reg_umap(Y, Y_h, y_r, embedding=None, X=None, size=10):
    # cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

    if embedding is None:
        import umap 
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(X)

    fig, axs = plt.subplots(1, 3, figsize=(21,5))
    data = np.hstack((embedding, Y.reshape(-1,1)))
    n_classes = len(list(set(Y.tolist())))

    df = pd.DataFrame(data=data, columns=['embd1', 'embd2', 'real'])
    sns.scatterplot(x='embd1', y='embd2', data=df, hue='real', ax=axs[0], palette="Set1", s=size)
    # sns.relplot(x='embd1', y='embd2', data=df, hue='real', ax=axs[0], palette="Set1", sizes=(10, 200))

    Yp = np.array(Y_h.reshape(-1,1), dtype=np.object)
    Yp[y_r < 0] = 'rejected'
    Yp[y_r > 0] = 'non-rejected'
    data = np.hstack((embedding, Yp))
    df = pd.DataFrame(data=data, columns=['embd1', 'embd2', 'rejection'])

    sns.scatterplot(x='embd1', y='embd2', data=df, hue='rejection', ax=axs[1], s=size, hue_order=['rejected', 'non-rejected'])
    # sns.relplot(x='embd1', y='embd2', data=df, hue='fit', ax=axs[0], palette="Set1", sizes=(10, 200))
    # plt.show()

    Yp = np.array(Y_h.reshape(-1,1), dtype=float)
    data = np.hstack((embedding[y_r > 0], Yp[y_r > 0]))
    df = pd.DataFrame(data=data, columns=['embd1', 'embd2', 'fit'])
    sns.scatterplot(x='embd1', y='embd2', data=df, hue='fit', ax=axs[2], s=size)
    # sns.relplot(x='embd1', y='embd2', data=df, hue='fit', ax=axs[0], palette="Set1", sizes=(10, 200))
    # plt.show()

def plot_selected_weight(pencil):
    w = pencil.gslayer.select_weight.detach().cpu().numpy()
    # plt.clf()
    plt.bar(np.arange(w.shape[0]), w)
    plt.show()

if __name__ == '__main__':
    cns = get_class_names('./datasets/Alarmin_MMUIL25_IL33/class_names.txt')



    
