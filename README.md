# GLOM
An attempted implementation of Geoffrey Hinton's paper "How to represent part-whole hierarchies in a neural network" for MNIST Dataset.

## Running
Open in jupyter notebook to run
Program expects an Nvidia graphics card for gpu speedup.

## Implementation details
Three Types of networks per number of vectors

1) Top-Down Network
2) Bottom-up Network
3) Attention on the same layer Network

Each network will see a 3x3 grid of vectors surrounding the current network input vector at the current layer. 
This is done to allow information to travel faster laterally across vectors. 

Since each network only sees a 3x3 grid and not larger image patches, this technique can be used for any size images and is parrallelizable. 

There is an initial state that all three types of network outputs get added to after every time step. 
The bottom layer of the state is the input vector where the MNIST pixel data is kept and doesn't get anything added to it to retain the MNIST pixel data.
The top layer of the state is the output layer where the loss function is applied to the networks. 

## Issues
There is a current issue that the networks will try and make all the output vectors the same. 
The source of the problem was thought to be caused by one of the reasons below, but no issues have been found so far:

* test putting MNIST data into first vector of the state is done correctly
* test if layers are being shifted down the correct dimensions
* test if concatening the rolls is done correctly
* test if networks are being added correctly
* test if output data is being shaped correclty for loss function
* test if output data is compared to the last vector of the state correctly

# If you find any issues, please feel free to contact me
