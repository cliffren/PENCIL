import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from numpy import size
from pencil import *
# from copy import deepcopy
# from memory_profiler import profile
# import inspect
# from gpu_mem_track import MemTracker
import tracemalloc
from time import time

data_name = 'rand-test'
expr_id = '02-search-c-cpu-only'

# @profile
def run_once(num_cells=50000, num_genes=2000, mode='multi-classification'):
    tracemalloc.start()
    tracemalloc.clear_traces()

    data = np.random.rand(num_cells, num_genes)
    labels = np.random.randint(0, 3, num_cells)

    pencil = Pencil(mode=mode, select_genes=True, seed=1234, data_name=data_name, expr_id=expr_id)
    # mlflow.end_run()
    mlflow.start_run()
    t0 = time()
    pred, confidence = pencil.fit_transform(
        data, labels, 
        test=False,
        # c=0.3, 
        cmax=1.0, cmin=0.0, 
        bisect_eps=0.05, shuffle_rate=1/3,
        lambda_L1=1e-5, 
        lambda_L2=1e-4, 
        lr=0.1, 
        epochs=500 + 500, 
        pre_train_epochs=0,
        class_weights=None,
        ) 
    t = time() - t0    
    mem = tracemalloc.get_traced_memory()
    print('peak memory : %.2f MiB, current memory : %.2f MiB' % (mem[1]/1024**2, mem[0]/1024**2))
    tracemalloc.stop()

    mlflow.end_run()

    return {'peak_memory(MiB)': mem[1]/1024**2, 'time(s)': t}
 
if __name__ == "__main__":

    num_cells_list = [1000, 2500, 5000, 10000, 25000, 50000, 100000, 150000, 200000, 250000]
    # num_cells_list = [1000000]
    # num_cells_list = [1000]
    # # num_cells_list = np.linspace(1000, 10000, 5).tolist()
    # num_cells_list = list(map(int, num_cells_list))

    df = pd.DataFrame()
    for mode in ['multi-classification', 'regression']:
        mems,  times = [], []
        for num_cells in num_cells_list:
            summary = run_once(num_cells=num_cells, num_genes=2000, mode=mode)
            mems.append(summary['peak_memory(MiB)'])
            times.append(summary['time(s)'])
        
        tmp_df = pd.DataFrame({
            'mode': [mode for _ in range(len(num_cells_list))],
            'num_of_cells': num_cells_list,
            'memory(MiB)': mems,
            'time(s)': times
        })
        df = pd.concat([df, tmp_df], axis=0)
    df.index = range(2 * len(num_cells_list))

    df.to_csv('./results/%s/py/%s/performance.csv' % (data_name, expr_id))
    
    df = pd.read_csv('./results/%s/py/%s/performance.csv' % (data_name, expr_id), index_col=0)

    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x='num_of_cells', y='time(s)', data=df, hue='mode', style='mode', ax=axs[0])
    sns.scatterplot(x='num_of_cells', y='memory(MiB)', data=df, hue='mode', style='mode', ax=axs[1])
    # sns.set(style='whitegrid')
    plt.savefig('./results/%s/py/%s/performance.pdf' % (data_name, expr_id))
    plt.show()









    