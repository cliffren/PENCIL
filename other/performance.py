from libs.dataloader import *
from libs.pencil import *
# from copy import deepcopy
# from memory_profiler import profile
# import inspect
# from gpu_mem_track import MemTracker
import tracemalloc

data_name = 'rand-test'
expr_id = '01'

#%%
def run_once(data, labels):
    pencil = Pencil(mode= 'multi-classification', select_genes=True, seed=1234, data_name=data_name, expr_id=expr_id)
    # mlflow.end_run()
    mlflow.start_run()
    pred, confidence = pencil.fit_transform(
        data, labels, 
        test=False,
        c=0.3, 
        lambda_L1=1e-5, 
        lambda_L2=1e-4, 
        lr=0.1, 
        epochs=100, 
        class_weights=None,
        )    
    mlflow.end_run()
    
    
#%%
if __name__ == "__main__":    
    tracemalloc.start()
    torch.cuda.reset_peak_memory_stats()
    num_cells, num_genes = 5000, 2000
    data = np.random.rand(num_cells, num_genes)
    labels = np.random.randint(0, 3, num_cells)

    run_once(data, labels)
    # print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('peak GPU memory : %.2f MiB, current GPU memory : %.2f MiB' % (torch.cuda.max_memory_allocated(0)/1024**2, torch.cuda.memory_allocated(0)/1024**2))
        
    mem = tracemalloc.get_traced_memory()
    print('peak memory : %.2f MiB, current memory : %.2f MiB' % (mem[1]/1024**2, mem[0]/1024**2))
    tracemalloc.stop()