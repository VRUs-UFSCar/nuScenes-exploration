import gc, shutil, torch, os

def clear_memory(cache_path):
    gc.collect()
    torch.cuda.empty_cache()

    if os.path.isdir(cache_path):
        shutil.rmtree(cache_path)