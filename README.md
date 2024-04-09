# Trabajo de final de grado de Jose Ángel Gallego García

### Preparación del workspace ###
# 1. Instalar: Python>=3.8 con PyTorch>=1.8
# 2. Crear un entorno virtual (Windows -> "python -m venv env"), y acceder a él
# 3. Instalar requeriments.txt (Windows -> "pip install -r requeriments.txt")

### Comando para entrenar una nueva versión del modelo ###
yolo task=segment mode=train epochs=100 data=data.yaml model=yolov8m.pt imgsz=640 batch= 4