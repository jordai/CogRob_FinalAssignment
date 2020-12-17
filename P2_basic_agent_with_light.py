import nengo


model = nengo.Network()

with model:
    

    # New inputs for the light and an ensemble that tells us if the light
    # is on or off
    lightInput = nengo.Node([0])
    light = nengo.Ensemble(500, 1)

    def threshold(x):
        if x > 0.8:
            return 1
        else:
            return 0

    nengo.Connection(lightInput, light, function=threshold)



    foodInput = nengo.Node([0,0])
    
    position = nengo.Ensemble(500, 2, radius=2)
    foodLocation = nengo.Ensemble(500,2, radius=2)
    
    nengo.Connection(foodInput, foodLocation)
    
    nengo.Connection(position, position) 

    #New ensemble to detremine which direction home is.
    #since home is the origin, the direction is given by our position
    #values encoded are 3Dsince we want to take the light info into account
    homeDiff = nengo.Ensemble(500, 3, radius=2)
    nengo.Connection(position, homeDiff[0:2], transform=-1)

    #Difference to food location, again with the option to take
    #light into account
    foodDiff = nengo.Ensemble(500,3, radius=2)

    nengo.Connection(position, foodDiff[0:2], transform=-1)
    nengo.Connection(foodLocation, foodDiff[0:2])
    
    #connect the light information to both diff ensembles
    nengo.Connection(light, foodDiff[2])
    nengo.Connection(light, homeDiff[2])
    
    #functions for how to use the light information
    
    #food: if light is on, go to food; else this should be 0
    def foodConnection(x):
        return x[0]*x[2], x[1]*x[2]

    #home: if light is off, go home; else this should be 0
    def homeConnection(x):
        return x[0]*(1-x[2]), x[1]*(1-x[2])

    #connect both direction options to posDiff using the functions
    #posDiff will now get (primarily) the difference to the location
    #we want to move to
    posDiff = nengo.Ensemble(500,2, radius=2)
    
    nengo.Connection(homeDiff, posDiff, function=homeConnection)
    nengo.Connection(foodDiff, posDiff, function=foodConnection)
    
    nengo.Connection(posDiff, position, transform= 0.1, synapse=0.5)
    
    #Exercise:
    #it is probably more likely that an agent knows about its home location
    #and the network above could easily be modified to use locaitons other
    #than (0,0) (how?)
    #
    #But, assuming home is always at (0,0), we could have found a simpler solution
    #which does not need the homeDiff ensemble purely by somehow inhibiting ONE
    #connection in the original model depending on the value from the light ensemble
    #Try to find and implement that solution.


