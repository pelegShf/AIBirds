# DQN Model for Angry Birds

We are team angryBIUrds.
This repository contains an implementation of a Deep Q-Network (DQN) model for playing Angry Birds. The DQN algorithm is used to learn an optimal strategy for playing the game through reinforcement learning.

## Usage

To use the DQN model for Angry Birds, run the main script with the following command:

- `<mode>`: Specify the mode ("train" or "test") in which the script should run.
- `--pre-trained-dir <path_to_directory>`: (Optional) Provide the directory containing pre-trained model weights for testing, or specify a directory for saving trained model checkpoints during training.

## Running the model:

1. Install Dependencies:
```bash
pip install -r requirements.txt
```
Note: Pytesserct can be troublesome, in consts.py change to the installation location.
- [Pytesserct install guide](https://pypi.org/project/pytesseract/)
- [Google tesserect install guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)


2. To train the DQN model:
```bash
python ./agent/main.py train
```

3. To Test the DQN model:
```bash
python ./agent/main.py test --pre-trained-dir /path/to/pretrained_model
```

## The project is structured as follows:

- **agent**: This directory contains the implementation of the reinforcement learning environment and agent.

- **external**: Here, you'll find the client-server implementation in Python for interacting with external services.

- **logs**: This directory holds saved logs generated during training or testing.

- **train**: In this directory, you'll find saved models resulting from training the DQN agent.

- **vision**: The vision directory contains the YOLOv8 model implementation for object detection.

Each of these directories contains the relevant code, data, and other resources associated with its respective component.

**IMPORTANT: The main file in this directory is not relevant and was used for dev only.**

