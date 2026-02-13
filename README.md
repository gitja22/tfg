# Trabajo de final de grado de Jose Ángel Gallego García

### Preparación del workspace ###
# 1. Instalar: Python>=3.8 con PyTorch>=1.8
# 2. Crear un entorno virtual (Windows -> "python -m venv env"), y acceder a él
# 3. Instalar requeriments.txt (Windows -> "pip install -r requeriments.txt")

### Comando para entrenar una nueva versión del modelo ###
yolo task=detect mode=train epochs=100 data=data.yaml model=yolov8x.pt imgsz=640 batch= 4

### comando para ejecutar la sección con contador - VÍDEO ###
python main.py --source "trafico_cruce.mp4" --view-img

### comando para ejecutar la sección con contador - CÁMARA ###
python main.py --source 0 --view-img --device 0
