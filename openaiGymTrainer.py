import argparse
import gym
import torch
from collections import OrderedDict
import copy
import os
import math
import numpy
import sys
import random
import ast

import NeuralNetworks.ConnectionStack


def main():
    print ("openaiGymTrainer.py main()")

    parser = argparse.ArgumentParser()
    parser.add_argument('OutputDirectory', help='The directory where the outputs will be written')
    parser.add_argument('OpenAIGymEnvironment', help='The openai gym environment')
    parser.add_argument('--testController', help='The filepath of a neural network to test. Default: None',
                        default=None)
    parser.add_argument('--hiddenLayerWidths', help="The tuple of hidden layers widths. Default:'(8, 5)'", default='(8, 5)')
    args = parser.parse_args()

    hiddenLayerWidths = ast.literal_eval(args.hiddenLayerWidths)

    env = gym.make(args.OpenAIGymEnvironment)
    # Extract action space data
    actionSpace = env.action_space
    if str(actionSpace).startswith('Discrete'):
        networkNumberOfOutputs = int(InsideOfParentheses(str(actionSpace)))
        actionSpaceLow = None
        actionSpaceHigh = None
    elif str(actionSpace).startswith('Box'):
        actionShapeStr = InsideOfParentheses(str(actionSpace))
        actionShape = [x.strip() for x in actionShapeStr.split(',') ]
        actionShape = [int(numberStr) for numberStr in actionShape if len(numberStr) > 0]
        if len(actionShape) > 1:
            raise NotImplementedError("openaiGymTrainer.py main(): The shape of action space ({}) is not supported".format(actionShape))
        networkNumberOfOutputs = actionShape[0]

        actionSpaceLow = torch.tensor(env.action_space.low)
        actionSpaceHigh = torch.tensor(env.action_space.high)
        # Make sure there is no 'infinite' value
        for index in range(actionSpaceLow.shape[0]):
            lowValue = actionSpaceLow[index].item()
            highValue = actionSpaceHigh[index].item()
            if lowValue < -1.0E6 or highValue > 1.0E6:
                actionSpaceLow[index] = 0
                actionSpaceHigh[index] = 1.0
    else:
        raise NotImplementedError("openaiGymTrainer.py main(): The action space '{}' is not supported".format(str(actionSpace)))

    # Extract observation space data
    observationSpace = env.observation_space
    if str(observationSpace).startswith('Discrete'):
        networkNumberOfInputs = int(InsideOfParentheses(str(observationSpace)))
        observationSpaceLow = None
        observationSpaceHigh = None
    elif str(observationSpace).startswith('Box'):
        observationShapeStr = InsideOfParentheses(str(observationSpace))
        observationShape = [x.strip() for x in observationShapeStr.split(',')]
        observationShape = [int(numberStr) for numberStr in observationShape if len(numberStr) > 0]
        if len(observationShape) > 1:
            raise NotImplementedError("openaiGymTrainer.py main(): The shape of observation space ({}) is not supported".format(observationShape))
        networkNumberOfInputs = observationShape[0]

        observationSpaceLow = torch.tensor(env.observation_space.low)
        observationSpaceHigh = torch.tensor(env.observation_space.high)
        # Make sure there is no 'infinite' value
        for index in range(observationSpaceLow.shape[0]):
            lowValue = observationSpaceLow[index].item()
            highValue = observationSpaceHigh[index].item()
            if lowValue < -1.0E6 or highValue > 1.0E6:
                observationSpaceLow[index] = 0
                observationSpaceHigh[index] = 1.0

    else:
        raise NotImplementedError("openaiGymTrainer.py main(): The observation space '{}' is not supported".format(str(observationSpace)))

    print ("main(): networkNumberOfInputs = {}; networkNumberOfOutputs = {}; actionSpaceLow = {}; actionSpaceHigh = {}; observationSpaceLow = {}; observationSpaceHigh = {}".format(
        networkNumberOfInputs, networkNumberOfOutputs, actionSpaceLow, actionSpaceHigh, observationSpaceLow, observationSpaceHigh
    ))



def InsideOfParentheses(stringToSearch):
    openingParentheseIndex = stringToSearch.index('(')
    closingParentheseIndex = stringToSearch.index(')')
    return stringToSearch[openingParentheseIndex + 1: closingParentheseIndex]


if __name__ == '__main__':
    main()