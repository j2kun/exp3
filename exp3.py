from probability import distr, draw
import math
import random

# exp3: int, (int, int -> float), float -> generator
# perform the exp3 algorithm.
# numActions is the number of actions, indexed from 0
# rewards is a function (or callable) accepting as input the action and
# producing as output the reward for that action
# gamma is an egalitarianism factor
def exp3(numActions, reward, gamma, rewardMin = 0, rewardMax = 1):
   weights = [1.0] * numActions

   t = 0
   while True:
      probabilityDistribution = distr(weights, gamma)
      choice = draw(probabilityDistribution)
      theReward = reward(choice, t)
      scaledReward = (theReward - rewardMin) / (rewardMax - rewardMin) # rewards scaled to 0,1

      estimatedReward = 1.0 * scaledReward / probabilityDistribution[choice]
      weights[choice] *= math.exp(estimatedReward * gamma / numActions) # important that we use estimated reward here!

      yield choice, theReward, estimatedReward, weights
      t = t + 1


# Test Exp3 using stochastic payoffs for 10 actions.
def simpleTest():
   numActions = 10
   numRounds = 10000

   biases = [1.0 / k for k in range(2,12)]
   rewardVector = [[1 if random.random() < bias else 0 for bias in biases] for _ in range(numRounds)]
   rewards = lambda choice, t: rewardVector[t][choice]

   bestAction = max(range(numActions), key=lambda action: sum([rewardVector[t][action] for t in range(numRounds)]))
   bestUpperBoundEstimate = 2 * numRounds / 3
   gamma = math.sqrt(numActions * math.log(numActions) / ((math.e - 1) * bestUpperBoundEstimate))
   gamma = 0.07

   cumulativeReward = 0
   bestActionCumulativeReward = 0
   weakRegret = 0

   t = 0
   for (choice, reward, est, weights) in exp3(numActions, rewards, gamma):
      cumulativeReward += reward
      bestActionCumulativeReward += rewardVector[t][bestAction]

      weakRegret = (bestActionCumulativeReward - cumulativeReward)
      regretBound = (math.e - 1) * gamma * bestActionCumulativeReward + (numActions * math.log(numActions)) / gamma

      print("regret: %d\tmaxRegret: %.2f\tweights: (%s)" % (weakRegret, regretBound, ', '.join(["%.3f" % weight for weight in distr(weights)])))

      t += 1
      if t >= numRounds:
         break

   print(cumulativeReward)


if __name__ == "__main__":
   simpleTest()
