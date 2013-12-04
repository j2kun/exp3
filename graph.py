from matplotlib import pyplot as plt
import numpy

def column(A, j):
    return [A[i][j] for i in range(len(A))]

def transpose(A):
    return [column(A, j) for j in range(len(A[0]))]


def regretWeightsGraph(filename, title):
   with open(filename, 'r') as infile:
      lines = infile.readlines()

   lines = [[eval(x.split(": ")[1]) for x in line.split('\t')] for line in lines]
   data = transpose(lines)

   regret = numpy.array(data[0])
   regretBound = numpy.array(data[1])
   weights = numpy.array(transpose(data[2]))
   xs = numpy.array(list(range(len(data[0]))))

   ax1 = plt.subplot(211)
   plt.ylabel('Cumulative (weak) Regret')
   ax1.plot(xs, regret)
   ax1.plot(xs, regretBound)
   plt.title(title)

   ax2 = plt.subplot(212)
   plt.ylabel('Weight')

   for w in weights:
      ax2.plot(xs, w)

   plt.show()


regretWeightsGraph('first-example.txt', "Regret of Exp3\n10 actions, 10k rounds, gamma = 0.07")

