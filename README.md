# Chrome Dinosaur AI

 Teaching AI to play a recreation of the Google Chrome dinosaur game. The game was made using pygame, and the AI part is done with NEAT-Python. 
 Watch a demo video [here](https://www.youtube.com/watch?v=TmwZNfGn--k)

# Using the application

 * Clone GitHub repository
 * Download required dependencies: `$ pip install -r requirements.txt`
 * `$ python main.py`
 
# How the Neural Network is setup:

 The neural network is configured with two inputs, one output, and 0 hidden layers.
 
 * Inputs: Distance from the dinosaur to the nearest cactus, the dinosaurs y position.
 * Outputs: Jump
 
 I have it setup to start with 15 dinosaurs per generation, however this can be changed by going into the config.txt and changing `pop_size`. Most of the values in config.txt are    default values. You can find a description of all of the settings [here](https://neat-python.readthedocs.io/en/latest/config_file.html)

# Contribution

  Any form of contribution is welcome! Please feel free to fork this project, modify/add to it, and make a pull request!

