
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights  # Essa rede foi escolhida por ser mais leve e mais rápida, não é igual ao faster-rcnn original
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes):
    # Carrega o modelo com os pesos pré-treinados
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    )
    
    # Altera a rede para que ela tenha o número de classes correto
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model