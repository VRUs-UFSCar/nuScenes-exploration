
Neste repositório estão os códigos de script que foram usados para o treinamento de modelos usando o PyTorch (mais especificamente os modelos de detecção de objetos do [torchvision](https://pytorch.org/vision/0.9/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)).

Na pasta `utils` estão vários arquivos com funções que podem ser usadas para o treinamento e validação de modelos do torchvision.

Em `train_faster_rcnn.py` está o código usado para o treinamento e validação do modelo Faster R-CNN. Modifique as configurações internas desse arquivo e o execute para treinar e validar este modelo.