### IMPORTS ###

import grid
import nengo
import numpy as np 
import nengo.spa as spa
import nengo.networks as networks


### CONSTANTS ###

MAP="""
#######
#  M  #
# # # #
# #B# #
#G Y R#
#######
"""

COLORS_TO_FIND = 4 # Number of colors to find before stopping (max. 5)

N_NEURONS = 50 # Number of neurons for Nengo ensembles
D = 32 # SPA state dimensionality

ROTATION_THRESHOLD = 0.8 # Threshold for random rotation in movement function (higher = less rotation)
STOP_SIM_THRESHOLD = 0.7 # Threshold for stopping (stop if similarity between memory and target exceeds this)


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
        return [body.detect(d, max_distance=4)[0] for d in angles]
    stim_radar = nengo.Node(detect)
    
    # Node for random values (filtered noise), to perform random rotations
    random_process = nengo.processes.FilteredNoise(dist=nengo.dists.Gaussian(0, 0.5), 
                                                   synapse=nengo.synapses.Alpha(0.1))
    random = nengo.Node(random_process)
    
    # Ensemble that reads sensor and random values
    radar = nengo.Ensemble(n_neurons=N_NEURONS*10, dimensions=4, radius=4)
    nengo.Connection(stim_radar, radar[0:3])
    nengo.Connection(random, radar[3])

    # Movement function, which outputs (speed, rotation) based on radar values
    def movement_func(x):
        left, forward, right, random = x
        # If random value exceeds thresholds, turn in the corresponding direction
        if abs(random) > ROTATION_THRESHOLD:
            rotation = abs(random) - forward/4
            return 0.1, rotation if random > 0 else -rotation
        # Otherwise, perform simple wall-avoiding behavior
        return forward/4, right - left
    
    # Movement function is driven only by radar values
    nengo.Connection(radar, movement, function=movement_func)
    
    
    ## COLOR DETECTION ##
    
    # Vocabulary of colors
    color_vocab = spa.Vocabulary(D, max_similarity=0)
    color_vocab.parse("NONE+GREEN+RED+BLUE+MAGENTA+YELLOW")
    
    # State that outputs the semantic pointer corresponding to the color of the currently occupied cell
    model.color_recognizer = spa.State(D, vocab=color_vocab)
    
    # Provide pointer corresponding to the color of the current cell as input to color recognizer
    def color_pointer(t):
        return body.cell.color().upper() if body.cell.color() else "NONE"
    model.current_color = spa.Input(color_recognizer=color_pointer)
    
    
    ## COLOR MEMORY ##
    
    # Vocabulary of booleans (true and false)
    bool_vocab = spa.Vocabulary(D, unitary=True)
    bool_vocab.add("FALSE", [1.]+[0.]*(D-1))
    bool_vocab.parse("TRUE")
   
    # Memories for all colors (supposed to store TRUE if color encountered, FALSE if not)
    model.green_memory   = spa.State(D, vocab=bool_vocab, label="green")
    model.red_memory     = spa.State(D, vocab=bool_vocab, label="red")
    model.blue_memory    = spa.State(D, vocab=bool_vocab, label="blue")
    model.magenta_memory = spa.State(D, vocab=bool_vocab, label="magenta")
    model.yellow_memory  = spa.State(D, vocab=bool_vocab, label="yellow")
    
    # Provide initial pointer "FALSE" to all color memories
    def initial_false_input(t):
        return bool_vocab["FALSE"].v.reshape(D) if t < 0.05 else np.zeros(D)
    false_input = nengo.Node(initial_false_input)
    nengo.Connection(false_input, model.green_memory.input)
    nengo.Connection(false_input, model.red_memory.input)
    nengo.Connection(false_input, model.blue_memory.input)
    nengo.Connection(false_input, model.yellow_memory.input)
    nengo.Connection(false_input, model.magenta_memory.input)
    
    # Cleanup memories for all color memories (to ensure that they store clean "boolean" pointers)
    model.green_cleanup   = spa.AssociativeMemory(bool_vocab, wta_output=True, label="clean G")
    model.red_cleanup     = spa.AssociativeMemory(bool_vocab, wta_output=True, label="clean R")
    model.blue_cleanup    = spa.AssociativeMemory(bool_vocab, wta_output=True, label="clean B")
    model.magenta_cleanup = spa.AssociativeMemory(bool_vocab, wta_output=True, label="clean M")
    model.yellow_cleanup  = spa.AssociativeMemory(bool_vocab, wta_output=True, label="clean Y")
    nengo.Connection(model.green_memory.output, model.green_cleanup.input, synapse=0.01)
    nengo.Connection(model.green_cleanup.output, model.green_memory.input, synapse=0.01)
    nengo.Connection(model.red_memory.output, model.red_cleanup.input, synapse=0.01)
    nengo.Connection(model.red_cleanup.output, model.red_memory.input, synapse=0.01)
    nengo.Connection(model.blue_memory.output, model.blue_cleanup.input, synapse=0.01)
    nengo.Connection(model.blue_cleanup.output, model.blue_memory.input, synapse=0.01)
    nengo.Connection(model.magenta_memory.output, model.magenta_cleanup.input, synapse=0.01)
    nengo.Connection(model.magenta_cleanup.output, model.magenta_memory.input, synapse=0.01)
    nengo.Connection(model.yellow_memory.output, model.yellow_cleanup.input, synapse=0.01)
    nengo.Connection(model.yellow_cleanup.output, model.yellow_memory.input, synapse=0.01)

    # Remove "FALSE" and add "TRUE" to a memory when its corresponding color is recognized
    actions = spa.Actions(
        'dot(color_recognizer, GREEN)   --> green_memory=TRUE-FALSE',
        'dot(color_recognizer, RED)     --> red_memory=TRUE-FALSE',
        'dot(color_recognizer, BLUE)    --> blue_memory=TRUE-FALSE',
        'dot(color_recognizer, MAGENTA) --> magenta_memory=TRUE-FALSE',
        'dot(color_recognizer, YELLOW)  --> yellow_memory=TRUE-FALSE',
        '0.5 --> '
    )
    model.basal_ganglia = spa.BasalGanglia(actions)
    model.thalamus = spa.Thalamus(model.basal_ganglia)
    
    
    ## COUNTING COLORS ##
    
    # Convolve all memories together
    model.cconv_gr    = networks.CircularConvolution(N_NEURONS, D, label="G*R")
    model.cconv_bm    = networks.CircularConvolution(N_NEURONS, D, label="B*M")
    model.cconv_bmy   = networks.CircularConvolution(N_NEURONS, D, label="B*M*Y")
    model.cconv_grbmy = networks.CircularConvolution(N_NEURONS, D, label="G*R*B*M*Y")
    nengo.Connection(model.green_memory.output, model.cconv_gr.A)
    nengo.Connection(model.red_memory.output, model.cconv_gr.B)
    nengo.Connection(model.blue_memory.output, model.cconv_bm.A)
    nengo.Connection(model.magenta_memory.output, model.cconv_bm.B)
    nengo.Connection(model.cconv_bm.output, model.cconv_bmy.A)
    nengo.Connection(model.yellow_memory.output, model.cconv_bmy.B)
    nengo.Connection(model.cconv_gr.output, model.cconv_grbmy.A)
    nengo.Connection(model.cconv_bmy.output, model.cconv_grbmy.B)
     
    # Specify what the convolved memory looks like when the desired number of colors has been encountered
    model.target = spa.State(D, vocab=bool_vocab)
    model.target_input = spa.Input(target = ("*TRUE"*COLORS_TO_FIND)[1:])
    
    # Compare desired and actual memory
    model.comparison = spa.Compare(D, vocab=bool_vocab)
    nengo.Connection(model.cconv_grbmy.output, model.comparison.inputA)
    nengo.Connection(model.target.output, model.comparison.inputB)
    
    
    ## STOPPING MOVEMENT ##
    
    # Extract the comparison value
    comparison = nengo.Ensemble(N_NEURONS,1)
    nengo.Connection(model.comparison.output, comparison)
   
    # Threshold the comparison value, to check if the agent is done
    done = nengo.Ensemble(N_NEURONS,1)
    nengo.Connection(comparison, done, function = lambda x: x > STOP_SIM_THRESHOLD)
    
    # Inhibitory connection between "being done" and "moving"
    nengo.Connection(done, radar.neurons, transform = [[-4]]*N_NEURONS*10)
    