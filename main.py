# importar librerías
from ultralytics import YOLO
import cv2

# leer modelo
model = YOLO("best2.pt")

# captura de video
cap = cv2.VideoCapture(0)

while True:
# leer frame
    ret, frame = cap.read()

    # leemos los resultados en tiempo real
    resultados = model.predict(frame, imgsz=640)

    # mostramos resultados
    anotaciones = resultados[0].plot()

    # detectar objetos
    cv2.imshow("Detección en tiempo real", anotaciones)

    # cerrar el programa
    if cv2.waitKey(1) == 27:
        break

# liberar recursos
cap.release()
cv2.destroyAllWindows()