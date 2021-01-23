### IMPORTS ###

import nengo
import nengo.spa as spa

D = 32
N_NEURONS = 50
    
COLORS_TO_FIND = 2

### SPA MODEL ###

with spa.SPA() as model:
    
   # Vocabulary for "boolean" pointers
   bool_vocab = spa.Vocabulary(D, unitary=True)
   bool_vocab.parse("TRUE")
   bool_vocab.add("FALSE", [1.] + [0.]*(D-1))
    
   # Memories for the five colors
   model.green_memory   = spa.State(D, vocab=bool_vocab, feedback=1, label="green")
   model.red_memory     = spa.State(D, vocab=bool_vocab, feedback=1, label="red")
   model.blue_memory    = spa.State(D, vocab=bool_vocab, feedback=1, label="blue")
   model.magenta_memory = spa.State(D, vocab=bool_vocab, feedback=1, label="magenta")
   model.yellow_memory  = spa.State(D, vocab=bool_vocab, feedback=1, label="yellow")
   
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
   model.seen_four_input = spa.Input(seen_four = ("*TRUE"*COLORS_TO_FIND)[1:])
   
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
   
    
    
    