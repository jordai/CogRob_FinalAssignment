import grid

mymap="""
#######
#  M  #
# # # #
# #B# #
#G Y R#
#######
"""

class Cell(grid.Cell):

    def color(self):
        if self.wall:
            return 'BLACK'
        elif self.cellcolor == 1:
            return 'GREEN'
        elif self.cellcolor == 2:
            return 'RED'
        elif self.cellcolor == 3:
            return 'BLUE'
        elif self.cellcolor == 4:
            return 'MAGENTA'
        elif self.cellcolor == 5:
            return 'YELLOW'
             
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
            
            
world = grid.World(Cell, map=mymap, directions=4)

body = grid.ContinuousAgent()
world.add(body, x=1, y=1, dir=2)

import nengo
import nengo.spa as spa
import numpy as np 

def move(t, x):
    speed, rotation = x
    dt = 0.001
    max_speed = 20.0
    max_rotate = 10.0
    body.turn(rotation * dt * max_rotate)
    body.go_forward(speed * dt * max_speed)


#Your model might not be a nengo.Netowrk() - SPA is permitted

model = spa.SPA()
with model:
    env = grid.GridNode(world, dt=0.005)
    
    movement = nengo.Node(move, size_in=2)
    
    #Three sensors for distance to the walls
    def detect(t):
        angles = (np.linspace(-0.5, 0.5, 3) + body.dir) % world.directions
        return [body.detect(d, max_distance=4)[0] for d in angles]
    stim_radar = nengo.Node(detect)
    
    radar = nengo.Ensemble(n_neurons=500, dimensions=3, radius=4)
    nengo.Connection(stim_radar, radar)
    #a basic movement function that just avoids walls based
    def movement_func(x):
        turn = x[2] - x[0]
        spd = x[1] - 0.5
        return spd, turn
    
    #the movement function is only driven by information from the
    #radar 
    nengo.Connection(radar, movement, function=movement_func)  
    
    #if you wanted to know the position in the world, this is how to do it
    #The first two dimensions are X,Y coordinates, the third is the orientation
    #(plotting XY value shows the first two dimensions)
    def position_func(t):
        return body.x / world.width * 2 - 1, 1 - body.y/world.height * 2, body.dir / world.directions
    
    position = nengo.Node(position_func)

    #This node returns the colour of the cell currently occupied. Note that you might want to transform this into 
    #something else (see the assignment)
    current_color = nengo.Node(lambda t:body.cell.cellcolor)
    
    # Set Dimension Size
    D = 32
    
    # Create Vocabulary
    colorVocab = spa.Vocabulary(D)
    
    #colorVocab.parse("GREEN+RED+BLUE+MAGENTA+YELLOW")
    colorVocab.parse("GREEN")
    
    # Function to recognize color:
    def recognizeColor(x, colorVocab = colorVocab, D=32):
        if x == 1:
            return colorVocab["GREEN"].v.reshape(D)
        elif x == 2:
            return colorVocab["RED"].v.reshape(D)
        elif x == 3:
            return colorVocab["BLUE"].v.reshape(D)
        elif x == 4:
            return colorVocab["MAGENTA"].v.reshape(D)
        elif x == 5:
            return colorVocab["YELLOW"].v.reshape(D)
        else:
            return np.zeros(D)
            
    # Create color SPA modules:
    model.colorRecognizer = spa.State(D, vocab = colorVocab)
    
    # Connect current_color to colorChecker:
    nengo.Connection(current_color, model.colorRecognizer.input, function = recognizeColor)
    
    # Create a memory:
    model.memory = spa.State(D, feedback=1, vocab = colorVocab, feedback_synapse = 0.4)

    nengo.Connection(model.colorRecognizer.output, model.memory.input)
    #nengo.Connection(model.memory.output, model.memory.input)