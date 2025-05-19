from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from pathlib import Path

# Inicialização do FastAPI
app = FastAPI()

# Caminho do modelo treinado
MODEL_PATH = "models/model_final.pth"

# Carregando o modelo Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Ajustar para o número de classes (1 para rachaduras)
cfg.MODEL.WEIGHTS = str(Path(MODEL_PATH))
cfg.MODEL.DEVICE = "cpu"  # Use "cuda" se tiver GPU configurada
predictor = DefaultPredictor(cfg)

# Rota de teste
@app.get("/")
def read_root():
    return {"message": "API Detectron2 está rodando corretamente!"}

# Rota para segmentação de imagem
@app.post("/segmenta)r")
async def segmentar_imagem(file: UploadFile = File(...)):
    try:
        # Lendo a imagem do arquivo
        image = await file.read()
        np_image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Realizando a predição
        outputs = predictor(image)

        # Processando os resultados
        instances = outputs["instances"].to("cpu")
        masks = instances.pred_masks.numpy()
        boxes = instances.pred_boxes.tensor.numpy()

        # Convertendo as máscaras para uma lista de listas
        masks_list = masks.tolist()

        return JSONResponse(content={"masks": masks_list, "boxes": boxes.tolist()})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
