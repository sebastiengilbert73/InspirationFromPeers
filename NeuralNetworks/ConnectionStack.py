import gym
import torch
from collections import OrderedDict
import copy
import os
import math
import numpy
import sys
import random

class NeuralNet(torch.nn.Module):
    def __init__(self, networkNumberOfInputs, hiddenLayerWidths, networkNumberOfOutputs,
                 actionSpace, observationSpace,
                 obervationLowTensor=None, observationHighTensor=None, actionLowTensor=None, actionHighTensor=None,
                 ):
        super(NeuralNet, self).__init__()
        layersDict = OrderedDict()
        for hiddenLayerNdx in range(len(hiddenLayerWidths) + 1):
            if hiddenLayerNdx == 0:
                numberOfInputs = networkNumberOfInputs # cartPoleEnv.observation_space
            else:
                numberOfInputs = hiddenLayerWidths[hiddenLayerNdx - 1]

            if hiddenLayerNdx == len(hiddenLayerWidths):
                numberOfOutputs = networkNumberOfOutputs
            else:
                numberOfOutputs = hiddenLayerWidths[hiddenLayerNdx]
            layersDict['layer' + str(hiddenLayerNdx)] = self.FullyConnectedLayer(numberOfInputs, numberOfOutputs)

        self.layers = torch.nn.Sequential(layersDict)
        self.apply(init_weights)
        self.actionSpace = actionSpace
        self.observationSpace = observationSpace
        self.observationLowTensor = obervationLowTensor
        self.observationHighTensor = observationHighTensor
        self.actionLowTensor = actionLowTensor
        self.actionHighTensor = actionHighTensor


    def forward(self, inputs):
        dataState = inputs
        for layerNdx in range(len(self.layers)):
            dataState = self.layers[layerNdx](dataState)
        if self.actionSpace.startswith('Discrete'):
            return torch.nn.functional.softmax(dataState, dim=0)
        elif self.actionSpace.startswith('Box'):
            return torch.sigmoid(dataState) # Squashes the value in ]0, 1[
        else:
            raise NotImplementedError("NeuralNet.forward(): Unimplemented action space '{}'".forward(self.actionSpace))

    def FullyConnectedLayer(self, numberOfInputs, numberOfOutputs):
        layer = torch.nn.Sequential(
            torch.nn.Linear(numberOfInputs, numberOfOutputs),
            torch.nn.ReLU()
        )
        return layer

    def act(self, observation, reward, done):
        inputTensor = torch.Tensor(self.RescaleObservation(observation))
        outputTensor = self.forward(inputTensor)
        if self.actionSpace.startswith('Discrete'):
            highestNdx = outputTensor.argmax().item()
            return highestNdx
        elif self.actionSpace.startswith('Box'):
            if self.actionLowTensor is not None and self.actionHighTensor is not None:
                return self.actionLowTensor + outputTensor * (self.actionHighTensor - self.actionLowTensor)
            else:
                return outputTensor
        else:
            raise NotImplementedError("NeuralNet.forward(): Unimplemented action space '{}'".forward(self.actionSpace))



    def PerturbateWeights(self, layerNdx, weightsDeltaSigma, biasDeltaSigma):
        if layerNdx < 0 or layerNdx >= len(self.layers):
            raise ValueError("NeuralNet.PerturbateWeights(): The layer index ({}) is out of the range [0, {}]".format(layerNdx, len(self.layers) - 1))
        weightsShape = self.layers[layerNdx][0].weight.shape
        biasShape = self.layers[layerNdx][0].bias.shape

        weightsDelta = torch.randn(weightsShape) * weightsDeltaSigma
        biasDelta = torch.randn(biasShape) * biasDeltaSigma

        self.layers[layerNdx][0].weight = torch.nn.Parameter(
            self.layers[layerNdx][0].weight + weightsDelta
        )
        self.layers[layerNdx][0].bias = torch.nn.Parameter(
            self.layers[layerNdx][0].bias + biasDelta
        )

    def PerturbateAllWeights(self, weightsDeltaSigma, biasDeltaSigma):
        for layerNdx in range(len(self.layers)):
            self.PerturbateWeights(layerNdx, weightsDeltaSigma, biasDeltaSigma)

    def Save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def Load(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=lambda storage, location: storage))

    def RescaleObservation(self, observation):
        return [ (observation[0] - self.observation_low[0])/(self.observation_high[0] - self.observation_low[0]), \
                 (observation[1] - self.observation_low[1]) / (self.observation_high[1] - self.observation_low[1]), \
                 (observation[2] - self.observation_low[2]) / (self.observation_high[2] - self.observation_low[2]), \
                 (observation[3] - self.observation_low[3]) / (self.observation_high[3] - self.observation_low[3]), \
                ]

    def MoveWeights(self, weightsDeltaList, biasDeltaList, learningRate):
        if len(weightsDeltaList) != len(self.layers) or len (biasDeltaList) != len(self.layers):
            raise ValueError("NeuralNet.MoveWeights(): The length weightsDeltaList({}) or the length of biasDeltaList ({}) doesn't equal the number of layers ({})".format(len(weightsDeltaList), len(biasDeltaList), len(self.layers)))
        for layerNdx in range(len(self.layers)):
            if weightsDeltaList[layerNdx].shape != self.layers[layerNdx][0].weight.shape:
                raise ValueError("NeuralNet.MoveWeights(): At index {}, the shape of the weightsDelta ({}) doesn't match the shape of the layer weights ({})".format(layerNdx, weightsDeltaList[layerNdx].shape, self.layers[layerNdx][0].weight.shape))
            self.layers[layerNdx][0].weight = torch.nn.Parameter(
                self.layers[layerNdx][0].weight + learningRate * weightsDeltaList[layerNdx]
            )

            if biasDeltaList[layerNdx].shape != self.layers[layerNdx][0].bias.shape:
                raise ValueError("NeuralNet.MoveWeights(): At index {}, the shape of the biasDelta ({}) doesn't match the shape of the layer bias ({})".format(
                        layerNdx, biasDeltaList[layerNdx].shape, self.layers[layerNdx][0].bias.shape))
            self.layers[layerNdx][0].bias = torch.nn.Parameter(
                self.layers[layerNdx][0].bias + learningRate * biasDeltaList[layerNdx]
            )
    def DeltasWith(self, otherNeuralNet):
        weightDeltasList = []
        biasDeltasList = []
        for layerNdx in range(len(self.layers)):
            if otherNeuralNet.layers[layerNdx][0].weight.shape != self.layers[layerNdx][0].weight.shape:
                raise ValueError("NeuralNet.DeltasWith(): For layer {}, the shape of the other neural net weight ({}) doesn't match mine ({})".format(layerNdx, otherNeuralNet.layers[layerNdx][0].weight.shape, self.layers[layerNdx][0].weight.shape))
            weightDelta = otherNeuralNet.layers[layerNdx][0].weight - self.layers[layerNdx][0].weight
            weightDeltasList.append(weightDelta)

            if otherNeuralNet.layers[layerNdx][0].bias.shape != self.layers[layerNdx][0].bias.shape:
                raise ValueError("NeuralNet.DeltasWith(): For layer {}, the shape of the other neural net bias ({}) doesn't match mine ({})".format(layerNdx, otherNeuralNet.layers[layerNdx][0].bias.shape, self.layers[layerNdx][0].bias.shape))
            biasDelta = otherNeuralNet.layers[layerNdx][0].bias - self.layers[layerNdx][0].bias
            biasDeltasList.append(biasDelta)
        return weightDeltasList, biasDeltasList


def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            #torch.nn.init.xavier_uniform_(m.bias)
