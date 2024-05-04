import sumo as sm
import traci as trci

sim = sm.Simulation.load("net/redPrueba.net.xml")
trci.init(port=8000)
vehicle_id = trci.vehicle.add("my_vehicle", "my_type", 0, 0, 0, 0, 0)
sim.close()
