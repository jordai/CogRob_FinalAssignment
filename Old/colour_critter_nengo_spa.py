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
    color_vocab = spa.Vocabulary(D, unitary=True)
    color_vocab.parse("GREEN+RED+BLUE+MAGENTA+YELLOW")
    
    # State that outputs the semantic pointer corresponding to the color of the currently occupied cell
    model.color_recognizer = spa.State(D, vocab=color_vocab)
    
    # Function for converting integers (0-5) to semantic pointers, using the color vocabulary
    def int_to_pointer(x):
        x = int(x)
        pointers = {1:"GREEN", 2:"RED", 3:"BLUE", 4:"MAGENTA", 5:"YELLOW"}
        return np.zeros(D) if not x else 10*color_vocab[pointers[x]].v.reshape(D)
    nengo.Connection(current_color, model.color_recognizer.input, function = int_to_pointer)

    model.color_memory = spa.State(D, vocab=color_vocab, feedback=1)

    color_checker = nengo.Ensemble(200,D*2,radius=10)
    
    nengo.Connection(model.color_recognizer.output, color_checker[0:D])
    nengo.Connection(model.color_memory.output, color_checker[D:D*2])
    
    def check_color(x):
        recognizer_state = x[0:D]
        memory_state = x[D:D*2]
        input_is_color = False
        for c in ["GREEN","RED","BLUE","MAGENTA","YELLOW"]:
            if np.dot(recognizer_state, color_vocab[c].v) > 0.8:
                if np.dot(memory_state, recognizer_state) < 0.3:
                    return 10*color_vocab[c].v

        return np.zeros(D)

    
    nengo.Connection(color_checker, model.color_memory.input, function=check_color)
    


    
    # # Memory state that keeps track of previously-encountered colors
    # model.color_memory = spa.State(D, vocab=color_vocab, feedback=1)
    # nengo.Connection(model.color_recognizer.output, model.color_memory.input)
    
    # Code for testing similarities with dot products
    # def green_similarity(x):
    #     out = np.dot(x, color_vocab["GREEN"].v)
    #     return out
    # how_green = nengo.Ensemble(50, 1)
    # nengo.Connection(mem, how_green, function=green_similarity)
    
    # # Code for getting vector magnitude
    # np.linalg.norm(vocab["GREEN"].v + vocab["GREEN"].v)
    
    # # Code for simulating outside of GUI
    # import matplotlib.pyplot as plt
    # p = nengo.Probe(model.color_memory.output)
    # with nengo.Simulator(model) as sim:
    #     sim.run(10)
    # plt.plot(sim.trange(), spa.similarity(sim.data[p], vocab)[:,0])
    # plt.show()
    
    # # Different way of detecting colors; unsure if this is more realistic or less realistic
    # def color_pointer(t):
    #     pointer = np.zeros(D) if not body.cell.color() else 10*color_vocab[body.cell.color().upper()].v.reshape(D)
    #     return pointer
    # current_color_pointer = nengo.Node(color_pointer)
    # nengo.Connection(current_color_pointer, model.color_memory.input)
    
    
    
    
    
    
    