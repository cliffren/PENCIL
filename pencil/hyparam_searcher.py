import numpy as np
import torch
from copy import deepcopy
from .pencil_trainer import train

def choose_value_of_c(pencil0, Xtr, Ytr, cmax, cmin=0.0, bisect_eps=0.05, shuffle_rate=1.0, check_thr=0.1, **kwgs):
    
    Ytr = deepcopy(Ytr)
    num_samples = Ytr.shape[0]
    select_ids = np.random.choice(num_samples, int(num_samples * shuffle_rate), replace=False)
    # if mode=='regression':
    #     Ytr[select_ids] = np.random.randn(len(select_ids))
    # else:
    labels_of_select_ids = Ytr[select_ids]
    np.random.shuffle(labels_of_select_ids)
    Ytr[select_ids] = labels_of_select_ids
    
    while cmax - cmin > bisect_eps:
        pencil = deepcopy(pencil0)
        c = (cmin + cmax) / 2
        pencil = train(
            pencil, Xtr, Ytr, 
            c=c, 
            use_lr_scheme=True, 
            plot_loss=False, 
            plot_show=False, 
            log_file=None, 
            silence=True,
            **kwgs
        )

        with torch.no_grad():
            if torch.cuda.is_available():
                _, r = pencil(torch.Tensor(Xtr).cuda())
            else:
                _, r = pencil(torch.Tensor(Xtr))
            r =  r.detach().cpu().numpy().flatten()
            # print('max', np.max(r))
            # print('mean', np.mean(r))
            # print(r)

        print('cmin:%.3f, cmax:%.3f, c:%.3f, rejected %d cells.' % (cmin, cmax, c, sum(r<0)))

        if sum(r>0) / r.shape[0] > check_thr:
            cmax = c
        else:
            cmin = c

        # print('cmin : cmax = %.3f : %.3f' % (cmin, cmax))

    return cmin