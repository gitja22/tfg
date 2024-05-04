import traci

# Connect to SUMO simulation
traci.start(["sumo", "-c", "net/flujoRutas.sumocfg"])

# Simulation loop
step = 0
while step < 1000:
    traci.simulationStep()
    # Your simulation logic here
    step += 1

# Close TraCI connection
traci.close()
