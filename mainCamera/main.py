#region librerias yolo
from __future__ import absolute_import
from __future__ import print_function


import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
#endregion

#region librerias sumo y traci
import os
import time
import sys
import random
import threading
#endregion

# importar los modulos de python del directorio $SUMO_HOME/tools
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa
#endregion

#region variables

#grupo de semaforos
grupo1 = 0
grupo2 = 0

ultimo_cambio_fase = time.time()

#endregion

track_history = defaultdict(list)
current_region = None
counting_regions = [
    {   
        # coordenadas pentagono -> Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)])
        "name": "region1",
        "polygon": Polygon([(200, 300), (440, 300), (440, 600), (200, 600)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (192, 57, 43),  # azul
        "text_color": (255, 255, 255),  # Region Text Color
    },
    {
        "name": "region2",
        "polygon": Polygon([(250, 250), (490, 250), (490, 550), (250, 550)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (244, 208, 63),  # cielo
        "text_color": (0, 0, 0),  # Region Text Color
    },
    {
        "name": "region3",
        "polygon": Polygon([(300, 250), (540, 250), (540, 550), (300, 550)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (39, 174, 96),  # verde
        "text_color": (0, 0, 0),  # Region Text Color
    },
    {
        "name": "region4",
        "polygon": Polygon([(350, 250), (590, 250), (590, 550), (350, 550)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (204, 0, 255),  # rosa
        "text_color": (0, 0, 0),  # Region Text Color
    },
]
#endregion

#region manipular raton
def mouse_callback(event, x, y, flags, param):
    global current_region
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False
#endregion

#region control semaforos
def control_semaforos():

    global grupo1
    global grupo2
    global ultimo_cambio_fase

    print("grupo1: "+ str(grupo1))
    print("grupo2: "+ str(grupo2))

    # Variables para controlar los semáforos
    tiempo_verde = 20  # Tiempo verde inicial
    tiempo_amarillo = 3  # Tiempo amarillo fijo
    tiempo_maximo = 60  # Tiempo máximo que una fase puede durar
    umbral = 3  # Diferencia mínima para cambiar la prioridad
    intervalo_minimo = 20  # Intervalo mínimo en segundos entre cambios de fase

    # Calcular la diferencia de carga entre los dos grupos
    diferencia = abs(grupo1 - grupo2)

    # Obtener el tiempo actual
    tiempo_actual = time.time()

    # Solo cambiar la fase si ha pasado el intervalo mínimo desde el último cambio
    if tiempo_actual - ultimo_cambio_fase >= intervalo_minimo:
        if grupo1 > grupo2 and diferencia > umbral:
            tiempo_verde = min(tiempo_verde + (diferencia // 2), tiempo_maximo)
            traci.trafficlight.setPhase("0", 0)  # Fase 0: arriba/abajo verde, izquierda/derecha rojo
            traci.trafficlight.setPhaseDuration("0", tiempo_verde)
            traci.trafficlight.setPhase("0", 1)  # Fase 1: arriba/abajo amarillo, izquierda/derecha rojo
            traci.trafficlight.setPhaseDuration("0", tiempo_amarillo)
            traci.trafficlight.setPhase("0", 4)  # Fase 4: ambos rojo para una pausa
            traci.trafficlight.setPhaseDuration("0", tiempo_amarillo)
        elif grupo2 > grupo1 and diferencia > umbral:
            tiempo_verde = min(tiempo_verde + (diferencia // 2), tiempo_maximo)
            traci.trafficlight.setPhase("0", 2)  # Fase 2: arriba/abajo rojo, izquierda/derecha verde
            traci.trafficlight.setPhaseDuration("0", tiempo_verde)
            traci.trafficlight.setPhase("0", 3)  # Fase 3: arriba/abajo rojo, izquierda/derecha amarillo
            traci.trafficlight.setPhaseDuration("0", tiempo_amarillo)
            traci.trafficlight.setPhase("0", 4)  # Fase 4: ambos rojo para una pausa
            traci.trafficlight.setPhaseDuration("0", tiempo_amarillo)
        else:
            # Fase de ciclo normal si las cargas son iguales o la diferencia es menor al umbral
            traci.trafficlight.setPhase("0", 0)
            traci.trafficlight.setPhaseDuration("0", tiempo_verde)
            traci.trafficlight.setPhase("0", 1)
            traci.trafficlight.setPhaseDuration("0", tiempo_amarillo)
            traci.trafficlight.setPhase("0", 2)
            traci.trafficlight.setPhaseDuration("0", tiempo_verde)
            traci.trafficlight.setPhase("0", 3)
            traci.trafficlight.setPhaseDuration("0", tiempo_amarillo)
        
        # Actualizar el tiempo del último cambio de fase
        ultimo_cambio_fase = tiempo_actual

#endregion

#region detección de objetos
def detect_objects(model, videocapture, line_thickness, track_thickness, region_thickness, view_img, classes):
    vid_frame_count = 0

    global grupo1
    global grupo2

    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        results = model.track(frame, persist=True, classes=classes)
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center
                track = track_history[track_id]  # Tracking Lines plot
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)
                
                # detectar dentro de las regiones
                for region in counting_regions:
                    if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        
                        # incrementar el contador visual
                        region["counts"] += 1

                        # incrementar el contador de grupo
                        if (region["name"] == "region1" or region["name"] == "region3"):
                            grupo1 = region["counts"]
                        elif (region["name"] == "region2" or region["name"] == "region4"):
                            grupo2 = region["counts"]                 
                     
        # dibujar las regiones
        for region in counting_regions:
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]
            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)
            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1,
            )
            cv2.putText(
                frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
            )
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)

        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Region Counter Movable")
                cv2.setMouseCallback("Region Counter Movable", mouse_callback)
            cv2.imshow("Region Counter Movable", frame)

        # resetear los contadores de regiones
        for region in counting_regions:
            region["counts"] = 0

        

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    del vid_frame_count
    videocapture.release()
    cv2.destroyAllWindows()
#endregion

#region funcion run
def run(
    weights="bestTrain.pt",
    source=None,
    device="0",
    view_img=False,
    save_img=False,
    exist_ok=False,
    classes=None,
    line_thickness=2,
    track_thickness=2,
    region_thickness=2,
):
    # configurar el modelo
    model = YOLO(f"{weights}")
    model.to("cuda") if device == "0" else model.to("cpu")

    global grupo1
    global grupo2


    # extraccion del nombre de las clases
    global names
    names = model.model.names

    # configurar el video
    videocapture = cv2.VideoCapture(0)

    detection_thread = threading.Thread(target=detect_objects, args=(model, videocapture, line_thickness, track_thickness, region_thickness, view_img, classes))
    detection_thread.start()

    step = 0
    #traci.trafficlight.setPhase("0", 2)

    # iterar la simulacion
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()     


        if grupo1 == 0 and grupo2 == 0:
            traci.trafficlight.setPhase("0", 4)
            traci.trafficlight.setPhaseDuration("0", 2000)         
        else:
            control_semaforos() 
        
        """
        FASES SEMAFORO
        if traci.trafficlight.getPhase("0") == 2:
            if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
                traci.trafficlight.setPhase("0", 3)
            else:
                traci.trafficlight.setPhase("0", 2)
        """
        step += 1

    traci.close()
    detection_thread.join()
#endregion

#region argumentos de la linea de comandos
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="bestTrain.pt", help="initial weights path")
    parser.add_argument("--device", default=0, help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--source", type=str, required=True, help="video file path")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--line-thickness", type=int, default=2, help="bounding box thickness (pixels)")
    parser.add_argument("--track-thickness", type=int, default=2, help="tracking line thickness (pixels)")
    parser.add_argument("--region-thickness", type=int, default=2, help="region line thickness (pixels)")
    opt = parser.parse_args()
    return opt
#endregion


#region generar archivo de rutas
def generate_routefile():
    random.seed(42)  # make tests reproducible
    N = 3600  # number of time steps
    # demand per second from different directions
    pWE = 1. / 10
    pEW = 1. / 11
    pNS = 1. / 30
    with open("data/cross.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeWE" color="white" imgFile="hyundai.png" accel="0.8" decel="4.5" sigma="0.5" length="10" height="5" width="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger" />
        <vType id="typeNS" color="white" imgFile="seat.png" accel="0.8" decel="4.5" sigma="0.5" length="10" height="5" width="7" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
        <vType id="typeSN" color="white" imgFile="seat.png" accel="0.8" decel="4.5" sigma="0.5" length="10" height="5" width="7" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
        <vType id="typeEW" color="white" imgFile="mg.png" accel="0.8" decel="4.5" sigma="0.5" length="10" height="5" width="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>

        <route id="right" edges="51o 1i 2o 52i" />
        <route id="left" edges="52o 2i 1o 51i" />
        <route id="down" edges="54o 4i 3o 53i" />""", file=routes)
        vehNr = 0
        for i in range(N):
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="right_%i" type="typeEW" route="right" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pEW:
                print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,1,1"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
        print("</routes>", file=routes)

#endregion

def main(opt):
    run(**vars(opt))

#region funcion main
if __name__ == "__main__":
    #recogida de los parámetros de yolo
    opt = parse_opt()

    #establecer sumo como servidor
    sumoBinary = checkBinary('sumo-gui')
    #generar ficha de ruta de sumo
    generate_routefile()

    #inicar traci 
    traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                             "--tripinfo-output", "tripinfo.xml"])

    main(opt)
#endregion
