# GLOM
An implementation of Geoffrey Hinton's paper "How to represent part-whole hierarchies in a neural network" for MNIST Dataset.
To understand this implementation, please watch [Yannick Kilcher's GLOM video](https://www.youtube.com/watch?v=cllFzkvrYmE&ab_channel=YannicKilcher), then read this README.md, then read the code. 

## Running
Open in jupyter notebook to run.
Program expects an Nvidia graphics card for gpu speedup.
If you run out of gpu memory, decrease the batch_size variable.
If you want to look at the code on github and it fails, try reloading or refreshing several times. 

## Results
The best models, which have been posted under the best_models folder, reached an accuracy of about ** 91% **.

## Implementation details
Three Types of networks per layer of vectors

1) Top-Down Network
2) Bottom-up Network
3) Attention on the same layer Network

### Intro to State
There is an initial state that all three types of network outputs get added to after every time step. 
The bottom layer of the state is the input vector where the MNIST pixel data is kept and doesn't get anything added to it to retain the MNIST pixel data.
The top layer of the state is the output layer where the loss function is applied and trained to be the one-hot MNIST target vector. 

### Explanation of compute_all function

Each type of network will see a 3x3 grid of vectors surrounding the current network input vector at the current layer. 
This is done to allow information to travel faster laterally across vectors, allowing for more information to be sent across an image in less steps. 
The easy way to do this is to shift (or roll) every vector along the x and y axis and then concatenate the vectors ontop of eachother so that every place a vector used to be in the state, now contains every vector and its neighboring vectors in the same layer. This also connects the edges of the image so that data can be passed from one edge of the image to the other, reducing the maximum distance any two pixels or vectors can be from one another. 

![](https://github.com/RedRyan111/GLOM/blob/main/Imgs/compute_all_function_concatenation.png)
For a more complex dataset, its possible this could pose some issues since two separate edges of an image aren't generally continous, but for MNIST, this problem doesn't arise. 
Then, these vectors are fed to each type of model. The models will get an input of all neighboring state vectors for a certain layer for each pixel that is given. Each model will then output a single vector. But there are 3 types of models per layer. In this example, every line drawn is a new model that is reused for every pixel this process is done for. After each model type has given an output, the three lists of vectors are added together. 

![](https://github.com/RedRyan111/GLOM/blob/main/Imgs/compute_all_function_concatenate_to_delta.png)

This will give a single list of vectors that will be added to the corresponding list of vectors at the specific x,y coordinate from the original state.

![](https://github.com/RedRyan111/GLOM/blob/main/Imgs/compute_all_function_add_delta_to_state.png)

Repeating this step for every list of vectors per x,y coordinate in the original state will yield the full new State value. 

Since each network only sees a 3x3 grid and not larger image patches, this technique can be used for any size images and is easily parrallelizable. 

## If I had more compute
My 2080Ti runs into memory errors running this if the batch size is above around 30, so here are my implementatin ideas if I had more compute. 
1) Increase batch_size. This probably wont affect the training, but it would make testing the accuracy faster.
2) Saving more states throughout the steps taken and adding them together. This would allow for gradients to get passed back to the original state similar to how a RESNET architecture can train very large models since the gradients can get passed backwards easier. This has been implemented to a smaller degree already and showed massive accuracy improvements.
3) Perform some kind of evolutionary parameter search by mutating the model parameters while also using backprop. This has been shown to improve the accuracy of image classifiers and other models. But this would take a ton of compute. 

## Yannic Kilcher's Attention
This has been pushed to github because during testing and tuning hyperparameters, a better model than previous was found. More testing needs to be done and I'm working on the visual explanation for it now. Previous versions of this code don't have the attention seen in the current version and will have similar performance. 

## Other Ideas behind the paper implementation
This is basically a neural cellular automata from the paper [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) with some inspiration from the follow up paper [Self-classifying MNIST Digits](https://distill.pub/2020/selforg/mnist/). Except instead of a single list of numbers (or one vector) per pixel, there are several vectors per pixel in each image. The Growing Neural Cellular Automata paper was very difficult to train also because the long gradient chains, so increasing the models complexity in this GLOM paper makes training even harder. But the neural cellular automata papers are the reason why the MSE loss function is used while also adding random noise to the state during training. 


### To do
1) Generated the explanation for [Yannick Kilcher's](https://www.youtube.com/channel/UCZHmQk67mSJgfCCTn7xBfew) version of attention that is implemented here. 
2) See if part-whole heirarchies are being found.
3) Keep testing hyperparameters to push accuracy higher.
4) Test different state initializations.
5) Train on harder datasets.
# If you find any issues, please feel free to contact me
