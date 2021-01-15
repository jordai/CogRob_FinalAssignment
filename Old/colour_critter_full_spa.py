### IMPORTS ###

import grid
import random
import nengo
import numpy as np 
import nengo.spa as spa
from nengo.networks.circularconvolution import circconv


### CONSTANTS ###

MAP="""
#######
#  M  #
# # # #
# #B# #
#G Y R#
#######
"""

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
    
    # Vocabulary of colors
    color_vocab = spa.Vocabulary(D, max_similarity=0, unitary=True)
    color_vocab.parse("GREEN+RED+BLUE+MAGENTA+YELLOW")
    
    # State that outputs the semantic pointer corresponding to the color of the currently occupied cell
    model.color_recognizer = spa.State(D, vocab=color_vocab)
    
    # Function for converting integers (0-5) to semantic pointers, using the color vocabulary
    def int_to_pointer(x):
        x = int(x)
        pointers = {1:"GREEN", 2:"RED", 3:"BLUE", 4:"MAGENTA", 5:"YELLOW"}
        return np.zeros(D) if not x else 10*color_vocab[pointers[x]].v.reshape(D)
    nengo.Connection(current_color, model.color_recognizer.input, function = int_to_pointer)
    
    bool_vocab = spa.Vocabulary(D, max_similarity=0, unitary=True)
    bool_vocab.parse("FALSE+TRUE")

    model.green_memory = spa.State(D,feedback=1)
    model.red_memory = spa.State(D,feedback=1)
    model.blue_memory = spa.State(D,feedback=1)
    model.magenta_memory = spa.State(D,feedback=1)
    model.yellow_memory = spa.State(D,feedback=1)

    actions = spa.Actions(
        'dot(color_recognizer, GREEN) -->   green_memory=10*TRUE',
        'dot(color_recognizer, RED) -->     red_memory=10*TRUE',
        'dot(color_recognizer, BLUE) -->    blue_memory=10*TRUE',
        'dot(color_recognizer, MAGENTA) --> magenta_memory=10*TRUE',
        'dot(color_recognizer, YELLOW) -->  yellow_memory=10*TRUE',
        '0.5 --> '
    )
    
    model.bg = spa.BasalGanglia(actions)
    model.th = spa.Thalamus(model.bg)
    
    seen_four = nengo.Ensemble(n_neurons=50, dimensions=1, radius=5)
    nengo.Connection(seen_four, radar.neurons, transform = [[-2]] * 500)

    def true_to_one(x):
        int_to_return = 1.0 if np.dot(x, bool_vocab["TRUE"].v) > 0.3 else 0.0
        return int_to_return
    
    
    
    