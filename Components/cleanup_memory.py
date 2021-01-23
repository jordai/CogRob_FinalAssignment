### IMPORTS ###

import nengo
import nengo.spa as spa
import numpy as np

D = 32
    


### SPA MODEL ###

with spa.SPA() as model:

  # Vocabulary for "boolean" pointers
  bool_vocab = spa.Vocabulary(D, unitary=True)
  bool_vocab.parse("TRUE")
  bool_vocab.add("FALSE", [1.] + [0.]*(D-1))
  
  # Memory for green color, including cleanup
  model.green_memory = spa.State(D, vocab=bool_vocab)
  model.green_cleanup = spa.AssociativeMemory(bool_vocab, wta_output=True)
  nengo.Connection(model.green_cleanup.output, model.green_memory.input, synapse=0.01)
  nengo.Connection(model.green_memory.output, model.green_cleanup.input, synapse=0.01)
  
  # Memory for red color, including cleanup
  model.red_memory = spa.State(D, vocab=bool_vocab)
  model.red_cleanup = spa.AssociativeMemory(bool_vocab, wta_output=True)
  nengo.Connection(model.red_cleanup.output, model.red_memory.input, synapse=0.01)
  nengo.Connection(model.red_memory.output, model.red_cleanup.input, synapse=0.01)
 
  # Function that provides the "FALSE" pointer as input at the start of the program
  def init_false_input(t):
      return bool_vocab["FALSE"].v.reshape(D) if t < 0.05 else np.zeros(D)
  
  # Node that initially provides the "FALSE" input       
  input = nengo.Node(init_false_input)
  nengo.Connection(input, model.green_memory.input) 
  nengo.Connection(input, model.red_memory.input)
  
  # "TRUE" input for the green memory
  green_input = nengo.Node(0)
  nengo.Connection(green_input, model.green_memory.input, 
                   function=lambda x: x*(bool_vocab["TRUE"].v-bool_vocab["FALSE"].v))
  
  # "TRUE" input for the red memory
  red_input = nengo.Node(0)
  nengo.Connection(red_input, model.red_memory.input, 
                   function=lambda x: x*(bool_vocab["TRUE"].v-bool_vocab["FALSE"].v))
  
  # Circular convolution between green and red memories
  model.cconv = nengo.networks.CircularConvolution(50,D)
  nengo.Connection(model.green_memory.output, model.cconv.A)
  nengo.Connection(model.red_memory.output, model.cconv.B)
  
  # "Desired" input, to compare circular convolution against
  model.desired = spa.State(D, vocab=bool_vocab)
  model.input = spa.Input(desired = "TRUE")
   
  # Comparison between circular convolution and desired input
  model.comparison = spa.Compare(D, vocab=bool_vocab)
  nengo.Connection(model.cconv.output, model.comparison.inputA)
  nengo.Connection(model.desired.output, model.comparison.inputB)
  
   