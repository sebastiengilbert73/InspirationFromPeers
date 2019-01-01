import argparse
import gym
import torch
import copy
import os
import math
import numpy
import sys
import random

class Population():
    def __init__(self,
                 individualsList,
                 # The individuals must implement:
                 #  individual.RandomMove
                 #  individual.MoveTowards(inspiringPeer, self.learningRate)
                 eliteProportion,
                 learningRate,
                 randomMoveProbability,
                 randomMoveStandardDeviationDic,
                 individualEvaluator, # Must implement: individualEvaluator.Evaluate(self.population)
                 printToConsole=True
                 ):
        self.population = individualsList
        self.eliteProportion = eliteProportion
        self.learningRate = learningRate
        self.randomMoveProbability = randomMoveProbability
        self.randomMoveStandardDeviationDic = randomMoveStandardDeviationDic
        self.individualEvaluator = individualEvaluator
        self.printToConsole = printToConsole
        self.individualToRewardDic = self.individualEvaluator.Evaluate(self.population)


    def EvolveOneCycle(self):
        averageReward, stdDevReward, maxReward = self.PopulationStatistics()
        if self.printToConsole:
            print ("averageReward = {}\t; stdDevReward = {}\t, maxReward = {}".format(averageReward, stdDevReward, maxReward))

        eliteList = self.Elite()

        nextPopulation = []
        for individual in self.population:
            # Decide if we'll do a random move or imitate an idol
            weDoARandomMove = numpy.random.random() < self.randomMoveProbability
            if weDoARandomMove:
                individual.RandomMove(self.randomMoveStandardDeviationDic)
                nextPopulation.append(individual)
            else:  # We are inspired by a peer
                inspiringPeer = random.choice(eliteList)
                # Move in the direction of the inspiring peer
                individual.MoveTowards(inspiringPeer, self.learningRate)
                nextPopulation.append(individual)
        self.population = nextPopulation
        self.individualToRewardDic = self.individualEvaluator.Evaluate(self.population)


    def PopulationStatistics(self):
        rewards = list(self.individualToRewardDic.values())
        average = numpy.mean(rewards)
        stdDev = numpy.std(rewards)
        maxReward = max(rewards)
        return average, stdDev, maxReward

    def Elite(self):
        numberOfEliteIndividuals = int(self.eliteProportion * len(self.individualToRewardDic))
        sortedPairsList = sorted(self.individualToRewardDic.items(), key=lambda kv: kv[1])  # Cf. https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
        sortedPairsList = sortedPairsList[len(
            sortedPairsList) - numberOfEliteIndividuals:]  # Keep the individuals with the highest reward
        return [individual for (individual, reward) in sortedPairsList]

    def Evaluate(self):
        return self.individualEvaluator.Evaluate(self.population)

    def Champion(self):
        sortedPairsList = sorted(self.individualToRewardDic.items(), key=lambda kv: kv[1])
        return sortedPairsList[-1][0], sortedPairsList[-1][1]