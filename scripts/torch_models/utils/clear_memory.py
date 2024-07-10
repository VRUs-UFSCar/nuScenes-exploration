import gc, shutil, torch, os

def clear_memory(cache_path):
    '''
    Faz uma limpeza da memória, tentando evitar que a memória da GPU fica sobrecarregada desenecessariamente.

    Args:
    - cache_path: str: caminho para a pasta de cache do pytorch. Deve estar no caminho "/home/<user>/.cache/torch", onde <user> é o nome do usuário do sistema operacional.
    '''
    gc.collect()
    torch.cuda.empty_cache()

    if os.path.isdir(cache_path):
        shutil.rmtree(cache_path)