from nuscenes.utils.splits import create_splits_scenes
from torch.utils.data import Dataset
import cv2
import os
import torch
from configs import RESIZE_PERCENT
from get_2D_boxes_from_sample_data import get_2D_boxes_from_sample_data


class NuScenesDataset(Dataset):
    def __init__(self, nusc_object, root_dir, data_type, classes, visibilities, return_tokens=False):
        '''
        Instancia a base de dados nuScenes para treinamento customizado. É preciso ter os dados da nuScenes no diretório `root_dir`.

        Args:
        - nusc_object: objeto NuScenes, criado a partir da biblioteca (devkit) da nuScenes.
        - root_dir: diretório onde os dados da nuScenes estão.
        - data_type: especifica quais cenas serão usadas (treino ou validação). É uma string podendo ser:
            - `mini_train`: cenas de treino do mini dataset.
            - `mini_val`: cenas de validação do mini dataset.
            - `train`: cenas de treino do dataset completo.
            - `val`: cenas de validação do dataset completo.
        - classes: lista de strings com os nomes das classes que devem ser detectadas. As classes devem ser as mesmas que estão no dataset da nuScenes.
        - visibilities: lista de strings com as visibilidades que devem ser consideradas. As visibilidades podem ser as seguintes strings: `'1', '2', '3', '4'`. A seguir estão as definições de cada visibilidade:
            - `1`: visibilidade entre 0 a 40% do objeto.
            - `2`: visibilidade entre 40 a 60% do objeto.
            - `3`: visibilidade entre 60 a 80% do objeto.
            - `4`: visibilidade entre 80 a 100% do objeto.
        - return_tokens: se `True`, retorna o token do sample data junto com a imagem e o target (útil para a validação, para coletar os bouding boxes originais - ground truth). Se `False`, retorna apenas a imagem e o target.
        '''
        # Salvando os parâmetros que precisarão ser usados depois
        self.root_dir = root_dir
        self.nusc_object = nusc_object
        self.data_type = data_type
        self.classes = classes
        self.visibilities = visibilities
        self.return_tokens = return_tokens
        
        # Coletando os tokens das cenas que serão usadas
        scene_names = create_splits_scenes()[data_type]
        scene_tokens = [nusc_object.field2token('scene', 'name', scene_name)[0] for scene_name in scene_names]

        # Coletando os sample datas de todas as cenas, para todas as câmeras
        self.samples_datas = []
        sensors = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        for scene_token in scene_tokens:
            scene = nusc_object.get('scene', scene_token)
            sample_token = scene['first_sample_token']
            while sample_token != '':
                sample = nusc_object.get('sample', sample_token)
                self.samples_datas.extend([nusc_object.get('sample_data', sample['data'][sensor]) for sensor in sensors])
                sample_token = sample['next']
        
        self.samples_datas = self.samples_datas[:20]

    def __getitem__(self, idx):
        '''
        Retorna a imagem, target e token do sample data (caso `return_tokens` seja `True`).
        Caso seja fornecido o índice (`idx`) 0, será retornado a primeira imagem da câmera frontal (da primeira cena do primeiro sample_data). O próximo idx será de uma outra câmera nesse mesmo sample_data. Quando todas as câmeras desse sample_data forem usadas, o próximo idx será do sample_data seguinte.

        Args:
        - idx: índice do sample data que se deseja carregar.
        '''
        # Coletando o sample data atual
        current_sample_data = self.samples_datas[idx]
        # Caminho da imagem
        image_path = os.path.join(self.root_dir, current_sample_data['filename'])

        # Lendo e "corrigindo" a imagem
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = cv2.resize(image, (int(image.shape[1] * RESIZE_PERCENT), int(image.shape[0] * RESIZE_PERCENT)))
        image = torch.as_tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)

        # Coletando as bounding boxes e labels
        boxes, labels = get_2D_boxes_from_sample_data(self.nusc_object, current_sample_data['token'], visibilities=self.visibilities)

        # Corrigindo as bounding boxes para o tamanho da imagem
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)  # (n, 4), importante para o treino ser nesse formato
        boxes *= RESIZE_PERCENT

        # Traduzindo os labels para os índices das classes
        for i in range(len(labels)):
            labels[i] = self.classes.index(labels[i])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Montando o target
        target = {
            'boxes': boxes,
            'labels': labels
        }

        if self.return_tokens:
            return image, target, current_sample_data['token']
        else:
            return image, target

    def __len__(self):
        return len(self.samples_datas)