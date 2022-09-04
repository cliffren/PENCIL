import torch 
import pandas as pd 
import numpy as np 

def simulated_data_1():
    d=1.50
    n=500
    X1=np.random.randn(n,2)+d*np.array([1,1])
    l1=[1.0]*len(X1)
    X2=np.random.randn(n,2)+d*np.array([1,-1])
    l2=[-1.0]*len(X2)
    X3=np.random.randn(n,2)+d*np.array([-1,1])
    l3=[-1.0]*len(X3)
    X4=np.random.randn(n,2)+d*np.array([-1,-1])
    l4=[1.0]*len(X4)
    data=np.vstack((X1, X2, X3, X4))
    labels=np.array(l1+l2+l3+l4)
    # pos=np.vstack((X1,X4))
    # neg=np.vstack((X2,X3))

    return data, labels

def simulated_data_0():
    d=0.8
    sigma = 1.2
    n=500
    X1=np.random.randn(n,2)*sigma + d*np.array([1,1])
    l1=[1.0]*len(X1)
    X2=np.random.randn(n,2)*sigma + d*np.array([-1,-1])
    l2=[-1.0]*len(X2)
    data=np.vstack((X1, X2))
    labels=np.array(l1+l2)
    # pos=np.vstack((X1,X4))
    # neg=np.vstack((X2,X3))

    return data, labels

def simulated_data_2(show=False, mix_rate_left=0.1, mix_rate_right=0.1):
    n_points = 500
    center_val = [0,-1]
    radisu_val = 1
    length = np.random.uniform(0, radisu_val, n_points)
    angle = np.pi * np.random.uniform(0, 2,n_points)
    x = np.sqrt(length) * np.cos(angle) + center_val[0]
    y = np.sqrt(length) * np.sin(angle) + center_val[1]
    A_X1= np.array((x,y))
    A_X1 = np.transpose(A_X1)
    condA_1=[1.0]*len(A_X1)
    clusterA_1=[3] * len(A_X1)

    center_val = [0,-1]
    radisu_val = 1
    length = np.random.uniform(0, radisu_val, n_points)
    angle = np.pi * np.random.uniform(0, 2,n_points)
    x = np.sqrt(length) * np.cos(angle) + center_val[0]
    y = np.sqrt(length) * np.sin(angle) + center_val[1]
    B_X1= np.array((x,y))
    B_X1 = np.transpose(B_X1)
    condB_1=[-1.0]*len(B_X1)
    clusterB_1=[3] * len(B_X1)

    center_val = [2,2]
    radisu_val = 1
    length = np.random.uniform(0, radisu_val, n_points)
    angle = np.pi * np.random.uniform(0, 2,n_points)
    x = np.sqrt(length) * np.cos(angle) + center_val[0]
    y = np.sqrt(length) * np.sin(angle) + center_val[1]
    A_X2= np.array((x,y))
    A_X2 = np.transpose(A_X2)
    condA_2=[1.0]*len(A_X2)
    num_mixed = int(n_points * mix_rate_right)
    condA_2 = np.array(condA_2)
    condA_2[0:num_mixed] = -1
    condA_2 = list(condA_2)
    clusterA_2=[2] * len(A_X2)


    center_val = [-2,2]
    radisu_val = 1
    length = np.random.uniform(0, radisu_val, n_points)
    angle = np.pi * np.random.uniform(0, 2,n_points)
    x = np.sqrt(length) * np.cos(angle) + center_val[0]
    y = np.sqrt(length) * np.sin(angle) + center_val[1]
    B_X2= np.array((x,y))
    B_X2 = np.transpose(B_X2)
    condB_2=[-1.0]*len(B_X2)
    num_mixed = int(n_points * mix_rate_left)
    condB_2 = np.array(condB_2)
    condB_2[0:num_mixed] = 1
    condB_2 = list(condB_2)
    clusterB_2=[1] * len(B_X2)

    data=np.vstack((A_X1, A_X2, B_X1, B_X2))
    labels=np.array(condA_1 + condA_2 + condB_1 + condB_2)
    clusters = np.array(clusterA_1 + clusterA_2 + clusterB_1 + clusterB_2)
    # pos=np.vstack((A_X1))
    # neg=np.vstack((B_X1))
    
    if show:
        plt.scatter(data[:,0],data[:,1],c=labels)
    
    return data, labels, clusters

def simulated_mul_clss_data():
    d=1.50
    n=500
    X1=np.random.randn(n,2)+d*np.array([1,1])
    l1=[0]*len(X1)
    X2=np.random.randn(n,2)+d*np.array([1,-1])
    l2=[1]*len(X2)
    X3=np.random.randn(n,2)+d*np.array([-1,1])
    l3=[2]*len(X3)
    X4=np.random.randn(n,2)+d*np.array([-1,-1])
    l4=[3]*len(X4)
    data=np.vstack((X1, X2, X3, X4))
    labels=np.array(l1+l2+l3+l4)
    return data, labels

def simulated_reg_data():
    d=1.50
    n=500
    X1=np.random.randn(n,2)+d*np.array([-2,1])
    l1=[0]*len(X1)
    X2=np.random.randn(n,2)+d*np.array([-1,2])
    l2=[1]*len(X2)
    X3=np.random.randn(n,2)+d*np.array([1,1])
    l3=[2]*len(X3)
    X4=np.random.randn(n,2)+d*np.array([2,-1])
    l4=[3]*len(X4)
    X5=np.random.randn(n,2)+d*np.array([4, -2])
    m1 = 10
    l5=[4]*(len(X5) - m1) + [-1] * m1

    X6=np.random.randn(n,2)+d*np.array([-3, -1])
    m2 = 50
    l6=[-1]*(len(X6) - m2) + [7] * m2
    data=np.vstack((X1, X2, X3, X4, X5, X6))
    labels=np.array(l1+l2+l3+l4+l5+l6)
    return data, labels

def load_real_data(
    exp_file='./datasets/Feldman_T_cell/Feldman_T_cell_scaled_data_exp.csv', 
    anno_file='./datasets/Feldman_T_cell/Feldman_T_cell_response_info.csv',
    class_names = ['Responder', 'Non-responder'], 
    mode = 'multi-classification'
    
): 
    # exp_file = './datasets/Feldman_T_cell/Feldman_T_cell_scaled_data_exp.csv'
    if exp_file is None:
        data = None
    else:
        exp_df=pd.read_csv(exp_file, sep=',',index_col=0)
        data=exp_df.values.T

    anno_df=pd.read_csv(anno_file, sep=',',index_col=0)
    anno_df = anno_df.values.flatten()
    
    if class_names is not None:
        if len(class_names) > 2 or mode == 'multi-classification':
            labels = label_encoder(anno_df, order=class_names)
        else:
            anno_df[anno_df==class_names[0]] = 1
            anno_df[anno_df==class_names[1]] = -1
            labels = anno_df.astype(float)
    else:
        labels = anno_df.astype(float)

    return data, labels

simulated_data = {
    'simulated_0': simulated_data_0,
    'simulated_1': simulated_data_1,
    'simulated_mul_class': simulated_mul_clss_data,
    'simulated_reg': simulated_reg_data
}

def label_encoder(x, order):
    label_dict = dict(zip(order, range(len(order))))
    labels = list(map(lambda s: label_dict[s], x))
    labels = np.array(labels)
    return labels

if __name__ == '__main__':
    data_name = 'GSE86028_TIL_MT1_KO'
    info_name = 'phenotype'
    select_genes_name = None
    # select_genes_name = 'deg1152'
    class_names = ['WT', 'MTKO']
    if select_genes_name is not None:
        exp_file='./datasets/%s/%s_scaled_data_exp_%s.csv' % (data_name, data_name, select_genes_name)
    else:
        exp_file='./datasets/%s/%s_scaled_data_exp.csv' % (data_name, data_name)

    anno_file='./datasets/%s/%s_%s_info.csv' % (data_name, data_name, info_name)

    # data, labels = load_real_data(exp_file, anno_file, class_names=class_names)

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    le = LabelEncoder()
    anno_df = pd.read_csv(anno_file, sep=',',index_col=0)
    anno_df = anno_df.values.flatten()
    
    labels = le.fit_transform(anno_df)
    labels_ = label_encoder(anno_df, order=class_names)

