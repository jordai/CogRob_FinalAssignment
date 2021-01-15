### IMPORTS ###

import nengo
import nengo.spa as spa
import numpy as np

D = 32
    


### SPA MODEL ###

with spa.SPA() as model:

  vocab = spa.Vocabulary(D, max_similarity=0, unitary=True)
  vocab.parse("FALSE+TRUE")
   
  model.green_memory = spa.State(D, vocab=vocab)
  model.green_cleanup = spa.AssociativeMemory(vocab, wta_output=True)
  nengo.Connection(model.green_cleanup.output, model.green_memory.input, synapse=0.01)
  nengo.Connection(model.green_memory.output, model.green_cleanup.input, synapse=0.01)
  
  model.red_memory = spa.State(D, vocab=vocab)
  model.red_cleanup = spa.AssociativeMemory(vocab, wta_output=True)
  nengo.Connection(model.red_cleanup.output, model.red_memory.input, synapse=0.01)
  nengo.Connection(model.red_memory.output, model.red_cleanup.input, synapse=0.01)
 
  
  def init_false_input(t):
      if t < 0.05:
          return vocab["FALSE"].v.reshape(D)
      else:
          return np.zeros(D)
          
          
  input = nengo.Node(init_false_input)
  nengo.Connection(input, model.green_memory.input) 
  nengo.Connection(input, model.red_memory.input)
  
  green_input = nengo.Node(0)
  nengo.Connection(green_input, model.green_memory.input, function=lambda x: x*(vocab["TRUE"].v-vocab["FALSE"].v))
  
  red_input = nengo.Node(0)
  nengo.Connection(red_input, model.red_memory.input, function=lambda x: x*(vocab["TRUE"].v-vocab["FALSE"].v))
  
  model.cconv = nengo.networks.CircularConvolution(50,D)
  nengo.Connection(model.green_memory.output, model.cconv.A)
  nengo.Connection(model.red_memory.output, model.cconv.B)
  
  model.comparison = spa.Compare(D, vocab=vocab)
  
  model.perfect = spa.State(D, vocab=vocab)
  model.input = spa.Input(perfect = "TRUE*FALSE")
  
  nengo.Connection(model.cconv.output, model.comparison.inputA)
  nengo.Connection(model.perfect.output, model.comparison.inputB)
    