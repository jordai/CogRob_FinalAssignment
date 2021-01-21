### IMPORTS ###

import grid
import random
import nengo
import numpy as np 
import nengo.spa as spa
import nengo.networks as networks


### CONSTANTS ###

MAP="""
#######
#  M  #
#   # #
#  B# #
#G Y R#
#######
"""

N_NEURONS = 50 # Number of neurons for Nengo ensembles
D = 32 # SPA state dimensionality
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

with spa.SPA() as model:
    
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
    radar = nengo.Ensemble(n_neurons=N_NEURONS*10, dimensions=3, radius=4)
    nengo.Connection(stim_radar, radar)

    # Movement function, which outputs (speed, rotation) based on radar values
    def movement_func(x):
        r = random.uniform(0,1)
        if r < 0.2:
            return 0.5, -0.7
        elif r < 0.4:
            return 0.5, 0.7
        else:
            return 0.5, ((x[2]-0.5)-(x[0]-0.5))/4
    
    # Movement function is driven only by radar values
    nengo.Connection(radar, movement, function=movement_func)
    
    # Inhibiting radar values
    done = nengo.Ensemble(N_NEURONS,1)
    done_slider = nengo.Node(0)
    nengo.Connection(done_slider, done)
    
    # Inhibitory connection between "being done" and "moving"
    nengo.Connection(done, radar.neurons, transform = [[-4]]*N_NEURONS*10)
    
    # Some testing with random processes (white noise)
    p = nengo.Node(nengo.processes.FilteredNoise(synapse=nengo.synapses.Alpha(0.1)))
    r = nengo.Ensemble(50,1)
    nengo.Connection(p,r)
    
    


    