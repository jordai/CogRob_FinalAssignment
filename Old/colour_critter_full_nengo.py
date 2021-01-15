### IMPORTS ###

import grid
import random
import nengo
import numpy as np 
import nengo.spa as spa
import nengo.networks.circularconvolution as circconv


### CONSTANTS ###

MAP="""
#######
#  M  #
# # # #
# #B# #
#G Y R#
#######
"""

D = 5 # SPA state dimensionality
MAX_DIST = 4 # Maximum distance of wall detectors


### CELL CLASS ###

class Cell(grid.Cell):

    def color(self):
        if self.wall:
            return 'black'
        elif self.cellcolor == 1:
            return 'green'
        elif self.cellcolor == 2:
            return 'red'
        elif self.cellcolor == 3:
            return 'blue'
        elif self.cellcolor == 4:
            return 'magenta'
        elif self.cellcolor == 5:
            return 'yellow'
        return None

    def load(self, char):
        self.cellcolor = 0
        if char == '#':
            self.wall = True
            
        if char == 'G':
            self.cellcolor = 1
        elif char == 'R':
            self.cellcolor = 2
        elif char == 'B':
            self.cellcolor = 3
        elif char == 'M':
            self.cellcolor = 4
        elif char == 'Y':
            self.cellcolor = 5


### INITIALIZING WORLD AND AGENT ###

world = grid.World(Cell, map=MAP, directions=4)
body = grid.ContinuousAgent()
world.add(body, x=1, y=2, dir=2)


### MOVEMENT FUNCTION ###

def move(t, x):
    speed, rotation = x
    dt = 0.001
    max_speed = 20.0
    max_rotate = 10.0
    body.turn(rotation * dt * max_rotate)
    body.go_forward(speed * dt * max_speed)
    


### SPA MODEL ###

with nengo.Network() as model:
    
    ## ENVIRONMENT INITIALIZATION ##
    
    # Initialize environment
    env = grid.GridNode(world, dt=0.005)
    
    
    ## MOVEMENT ##
    
    # Node that handles agent movement (input is (speed, rotation))
    movement = nengo.Node(move, size_in=2)
    
    # Node for the three wall distance sensors
    def detect(t):
        # Define angles for each detector (left, forward, right)
        angles = (np.linspace(-0.5, 0.5, 3) + body.dir) % world.directions
        # Return the distance between the agent and a wall in the given directions
        return [body.detect(d, max_distance=MAX_DIST)[0] for d in angles]
    stim_radar = nengo.Node(detect)
    
    # Ensemble that reads sensor values
    radar = nengo.Ensemble(n_neurons=500, dimensions=3, radius=4)
    nengo.Connection(stim_radar, radar)

    # Movement function, which outputs (speed, rotation) based on radar values
    def movement_func(x):
        spd = (x[1] - 0.5)*0.3
        if x[1] < 0.5:
            return spd, 1
        if x[0] < 0.5:
            return spd, random.uniform(0,30)
        if x[2] < 0.5:
            return spd, random.uniform(-30,0)
        return spd, 0
    
    # Movement function is driven only by radar values
    nengo.Connection(radar, movement, function=movement_func)
    
    
    ## POSITION (currently not used) ##
    
    # # Function for extracting agent position (returns (X,Y,orientation))
    # def position_func(t):
    #     return body.x / world.width * 2 - 1, 1 - body.y/world.height * 2, body.dir / world.directions
    # # Node that keeps track of agent position
    # position = nengo.Node(position_func)
    
    
    ## COLOR ##

    # Node that outputs the color of the currently occupied cell
    current_color = nengo.Node(lambda t:body.cell.cellcolor)
    
    color_recognizer = nengo.Ensemble(1000, dimensions=1, radius=5)
    nengo.Connection(current_color, color_recognizer)
    
    color_checker = nengo.Ensemble(1000, dimensions=6, radius=5)
    nengo.Connection(color_recognizer, color_checker[0])
    
    color_memory = nengo.Ensemble(500, dimensions=5, radius=6)
    nengo.Connection(color_memory, color_checker[1:6])
    
    def check_color(x):
        current_color = round(x[0])
        print(current_color)
        return [0,0,0,0,0]
    nengo.Connection(color_checker, color_memory, function=check_color)
    
    ## MOVEMENT INHIBITION ##
    
    # Inhibitory connection between node that represents having seen 4 colors and radar (to inhibit movement)
    seen_four = nengo.Ensemble(50,1)
    nengo.Connection(seen_four, radar.neurons, transform = [[-2]] * 500)
    # Temporary slider to test inhibition
    seen_four_input = nengo.Node(0, label="")
    nengo.Connection(seen_four_input, seen_four)
    
   
    import matplotlib.pyplot as plt
    p = nengo.Probe(color_recognizer)
    with nengo.Simulator(model) as sim:
        sim.run(10)
    plt.plot(sim.trange(), sim.data[p])
    plt.show()
    

    
    
    
    
    