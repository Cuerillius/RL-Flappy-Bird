# Reinforcement Learning Flappy Bird

This repository contains a simple implementation of a reinforcement learning agent that plays the Flappy Bird game. The agent is trained using the Q-learning algorithm.

## Training the Agent

To train the reinforcement learning agent, run the following command:

```bash
python rl.py
```

This will start the training process, and the agent will learn to play the game over time.

The training saves the model weights to a numpy file named `q_table.npy` after training is complete. You can adjust the training parameters in the `rl.py` script as needed.

## Thanks

This implementation is built on top of the Flappy Bird game code from [this repository](https://github.com/mehmetemineker/flappy-bird).
