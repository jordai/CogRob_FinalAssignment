### IMPORTS ###

import nengo
import nengo.spa as spa

D = 32
N_NEURONS = 50
    


### SPA MODEL ###

with spa.SPA() as model:
    
   # Vocabulary for "boolean" pointers
   bool_vocab = spa.Vocabulary(D)
   bool_vocab.parse("FALSE+TRUE")
    
   # Memories for the five colors
   model.green_memory   = spa.State(D, vocab=bool_vocab)
   model.red_memory     = spa.State(D, vocab=bool_vocab)
   model.blue_memory    = spa.State(D, vocab=bool_vocab)
   model.magenta_memory = spa.State(D, vocab=bool_vocab)
   model.yellow_memory  = spa.State(D, vocab=bool_vocab)
   
   # Circular convolutions that combine the color memories
   model.gr_conv    = nengo.networks.CircularConvolution(N_NEURONS,D,label="G*R")
   model.bm_conv    = nengo.networks.CircularConvolution(N_NEURONS,D,label="B*M")
   model.bmy_conv   = nengo.networks.CircularConvolution(N_NEURONS,D,label="B*M*Y")
   model.grbmy_conv = nengo.networks.CircularConvolution(N_NEURONS,D,label="G*R*B*M*Y")
   
   # Combining green and red
   nengo.Connection(model.green_memory.output, model.gr_conv.A)
   nengo.Connection(model.red_memory.output, model.gr_conv.B)
   
   # Combining blue and magenta
   nengo.Connection(model.blue_memory.output, model.bm_conv.A)
   nengo.Connection(model.magenta_memory.output, model.bm_conv.B)
   
   # Combining yellow and blue+magenta
   nengo.Connection(model.yellow_memory.output, model.bmy_conv.A)
   nengo.Connection(model.bm_conv.output, model.bmy_conv.B)
   
   # Combining red+green and blue+magenta+yellow
   nengo.Connection(model.gr_conv.output, model.grbmy_conv.A)
   nengo.Connection(model.bmy_conv.output, model.grbmy_conv.B)
   
   # Specify what the vector looks like after seeing four colors
   model.seen_four = spa.State(D, vocab=bool_vocab)
   model.seen_four_input = spa.Input(seen_four = "TRUE*TRUE*TRUE*TRUE*FALSE")
   
   # Compare perfect "seen four" vector to combination of memories
   model.comparison = spa.Compare(D, vocab=bool_vocab)
   nengo.Connection(model.grbmy_conv.output, model.comparison.inputA)
   nengo.Connection(model.seen_four.output, model.comparison.inputB)
   
   # Extract the comparison value
   comp_value = nengo.Ensemble(N_NEURONS,1)
   nengo.Connection(model.comparison.output, comp_value)
   
   # Threshold the comparison value, to check if the agent is done
   done = nengo.Ensemble(N_NEURONS,1)
   nengo.Connection(comp_value, done, function = lambda x: x > 0.9)
   
    
    
    