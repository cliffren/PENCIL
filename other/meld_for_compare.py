
import os

from libs.dataloader import *
from libs.utils import *
from copy import deepcopy

import meld
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy

setup_seed(1234) #fix the random state.

################# select dataset #######################################
## set the dataset name and expriment ID for recording the results.
## data_name: simulated_1 | simulated_2 | Feldman_T_cell | ...
## see `dataset_config_template.py` for more configs.
# simulate data for multi-class.
data_name = 'Feldman_T_cell_for_select_genes_biclassification_compare2'
class_names = ['class_1', 'class_2']
save_path = './results/%s/meld' % data_name
exp_file ='./datasets/%s/exp_data.csv' % (data_name)
try:
    os.makedirs(save_path)
except:
    pass

for  mix_rate in [0.0, 0.1, 0.2, 0.3]:
    for gene_select_id in range(18):
        anno_file ='./datasets/%s/label_info_id%d_mix%.1f.csv' % (data_name, gene_select_id, mix_rate)
        expr_id = 'id%d_mix%.1f' % (gene_select_id, mix_rate) #This will be included in the output-file. Anything else can be recorded here. 

        embedding_name = 'embedding_id%d_mix%.1f' % (gene_select_id, mix_rate)
        emd = pd.read_csv('./datasets/%s/%s.csv' % (data_name, embedding_name), index_col=0).values

        class_weights = None
        print('dataset: %s, expr_id: %s' % (data_name, expr_id))


        ####################### load data ################################################
        data, labels = load_real_data(exp_file, anno_file, class_names=class_names)
        labels = np.array(class_names)[labels]
        print(labels)

        # Estimate density of each sample over the graph
        sample_densities = meld.MELD().fit_transform(data[:,:], labels)
        # Normalize densities to calculate sample likelihoods
        sample_likelihoods = meld.utils.normalize_densities(sample_densities)

        num_classes = len(class_names)
        mixture_model = GaussianMixture(n_components=num_classes + 1)
        classes = mixture_model.fit_predict(sample_likelihoods.values)

        mean_entropy = lambda c : entropy(sample_likelihoods.values[classes==c].T, base=num_classes).mean()
        entropys = np.round(list(map(mean_entropy, range(num_classes + 1))), 2)
        predict = np.array(class_names, dtype='object')[np.argmax(sample_likelihoods.values, axis=1)]

        for i in range(num_classes + 1):
            if entropys[i] == 1.0:
                predict[classes==i] = 'meld-rejected'

        rst = pd.DataFrame()

        rst = deepcopy(sample_likelihoods)
        rst['labels'] = labels
        rst['predict'] = predict
        rst['emd1'] = emd[:,0]
        rst['emd2'] = emd[:,1]

        # fig, axs = plt.subplots(1, 2, figsize=(12,5))
        # sns.scatterplot(x='emd1', y='emd2', hue='labels', data=rst, s=10, ax=axs[0])
        # sns.scatterplot(x='emd1', y='emd2', hue='predict', data=rst, s=10, ax=axs[1])
        # plt.show()

        del rst['emd1']
        del rst['emd2']
        rst.to_csv('%s/meld_results_%s.csv' % (save_path, expr_id))
    
    

            
            