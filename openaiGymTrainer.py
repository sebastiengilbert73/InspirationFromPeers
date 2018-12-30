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
import InspirationFromPeers


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


    if args.testController is not None:
        agent = NeuralNetworks.ConnectionStack.NeuralNet(
            networkNumberOfInputs, hiddenLayerWidths, networkNumberOfOutputs,
            str(actionSpace), str(observationSpace),
            observationSpaceLow, observationSpaceHigh,
            actionSpaceLow, actionSpaceHigh
        )
        agent.Load(args.testController)
        rewardSumsList = []
        for i_episode in range(10):
            observation = env.reset()
            rewardSum = 0
            done = False
            while not done:
                env.render()
                print (observation)
                reward = 0
                done = False
                action = agent.act(observation, reward, done)
                #action = env.action_space.sample() # Random choice
                observation, reward, done, info = env.step(action)
                rewardSum += reward
                if done:
                    print ("Breaking! rewardSum = {}".format(rewardSum))
                    break
            rewardSumsList.append(rewardSum)
        print ("main(): rewardSumsList: {}".format(rewardSumsList))
        averageReward, highestReward = TournamentStatistics(rewardSumsList)
        print ("main(): averageReward = {}; highestReward = {}".format(averageReward, highestReward))
        sys.exit()

    evaluator = Evaluator(environment=env, numberOfEpisodesForEvaluation=30)

    numberOfIndividuals = 15
    eliteProportion = 0.1
    learningRate = 0.1
    randomMoveProbability = 0.2
    randomMoveStandardDeviationDic = {'weight': 1.0, 'bias': 0.3}
    numberOfCycles = 10

    individualsList = []
    for individualNdx in range(numberOfIndividuals):
        individualsList.append(NeuralNetworks.ConnectionStack.NeuralNet(
            networkNumberOfInputs, hiddenLayerWidths, networkNumberOfOutputs,
            str(actionSpace), str(observationSpace),
            observationSpaceLow, observationSpaceHigh,
            actionSpaceLow, actionSpaceHigh
        ))

    population = InspirationFromPeers.Population(
        individualsList=individualsList,
        eliteProportion=eliteProportion,
        learningRate=learningRate,
        randomMoveProbability=randomMoveProbability,
        randomMoveStandardDeviationDic=randomMoveStandardDeviationDic,
        individualEvaluator=evaluator
    )

    with open(os.path.join(args.OutputDirectory, 'stats.csv'), "w+") as statsFile:
        statsFile.write("Cycle,AverageReward,RewardStandardDeviation,CurrentPopulationHighestReward,ChampionReward\n")
    highestReward = -sys.float_info.max
    champion = NeuralNetworks.ConnectionStack.NeuralNet(
            networkNumberOfInputs, hiddenLayerWidths, networkNumberOfOutputs,
            str(actionSpace), str(observationSpace),
            observationSpaceLow, observationSpaceHigh,
            actionSpaceLow, actionSpaceHigh
        )

    for cycleNdx in range(numberOfCycles):
        print ("\n --- Cycle {} ---".format(cycleNdx + 1))
        population.EvolveOneCycle()
        averageReward, stdDevReward, maxReward = population.PopulationStatistics()

        with open(os.path.join(args.OutputDirectory, 'stats.csv'), "a+") as statsFile:
            statsFile.write(str(cycleNdx + 1) + ',' + str(averageReward) + ',' + str(stdDevReward) + ',' + str(maxReward) + ',' + str(highestReward) + '\n')

        if maxReward > highestReward:
            highestReward = maxReward
            populationChampion, _ = population.Champion()
            champion = copy.deepcopy(populationChampion)
            champion.Save(os.path.join(args.OutputDirectory, \
                                              'champion_' + str(hiddenLayerWidths) + '_' + str(highestReward)))

def InsideOfParentheses(stringToSearch):
    openingParentheseIndex = stringToSearch.index('(')
    closingParentheseIndex = stringToSearch.index(')')
    return stringToSearch[openingParentheseIndex + 1: closingParentheseIndex]

def TournamentStatistics(tournamentAverageRewards):
    if len(tournamentAverageRewards) == 0:
        raise ValueError("TournamentStatistics(): The input list is empty")
    highestReward = -sys.float_info.max
    rewardsSum = 0
    for reward in tournamentAverageRewards:
        if reward > highestReward:
            highestReward = reward
        rewardsSum += reward
    return rewardsSum / len(tournamentAverageRewards), highestReward

class Evaluator():
    def __init__(self, environment, numberOfEpisodesForEvaluation):
        self.environment = environment
        self.numberOfEpisodesForEvaluation = numberOfEpisodesForEvaluation

    def Evaluate(self, population):
        individualToRewardDic = {}
        for individual in population:
            individualRewardSum = 0
            for episodeNdx in range(self.numberOfEpisodesForEvaluation):
                observation = self.environment.reset()
                episodeRewardSum = 0
                done = False
                actionReward = 0
                while not done:
                    action = individual.act(observation, actionReward, done)  # Choose an action
                    observation, actionReward, done, info = self.environment.step(action)  # Perform the action
                    episodeRewardSum += actionReward
                    if done:
                        break
                individualRewardSum += episodeRewardSum
            individualToRewardDic[individual] = individualRewardSum / self.numberOfEpisodesForEvaluation
        return individualToRewardDic



if __name__ == '__main__':
    main()