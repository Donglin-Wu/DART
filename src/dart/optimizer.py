"""
粒子群优化。
"""

import numpy as np


class PSO:
    def __init__(self, objFunc, bounds, popSize=50, iters=30, w=0.729, a1=1.49445, a2=1.49445, randomState=None):
        self.objFunc = objFunc
        self.bounds = np.array(bounds, dtype=float)
        self.popSize = popSize
        self.iters = iters
        self.w = w
        self.a1 = a1
        self.a2 = a2
        self.rng = np.random.RandomState(randomState)
        self.dim = len(bounds)

        self.pos = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(popSize, self.dim))
        velocityScale = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.2
        self.vel = self.rng.uniform(-velocityScale, velocityScale, size=(popSize, self.dim))

        self.pbest = self.pos.copy()
        self.pbestVal = np.array([self.objFunc(point) for point in self.pos], dtype=float)
        self.gbest = self.pbest[np.argmax(self.pbestVal)].copy()
        self.gbestVal = float(np.max(self.pbestVal))

    def Optimize(self):
        for _ in range(self.iters):
            randFirst = self.rng.rand(self.popSize, self.dim)
            randSecond = self.rng.rand(self.popSize, self.dim)
            self.vel = (
                self.w * self.vel
                + self.a1 * randFirst * (self.pbest - self.pos)
                + self.a2 * randSecond * (self.gbest - self.pos)
            )

            self.pos += self.vel
            self.pos = np.minimum(np.maximum(self.pos, self.bounds[:, 0]), self.bounds[:, 1])
            values = np.array([self.objFunc(point) for point in self.pos], dtype=float)

            improvedMask = values > self.pbestVal
            if np.any(improvedMask):
                self.pbest[improvedMask] = self.pos[improvedMask]
                self.pbestVal[improvedMask] = values[improvedMask]

            bestIndex = int(np.argmax(self.pbestVal))
            if self.pbestVal[bestIndex] > self.gbestVal:
                self.gbestVal = float(self.pbestVal[bestIndex])
                self.gbest = self.pbest[bestIndex].copy()

        return self.gbest, self.gbestVal


def PsoSearchExtrema(modelFunc, bounds, searchCount=5, popSize=None, iters=30, randomState=None):
    dimension = len(bounds)
    if popSize is None:
        popSize = 5 * dimension

    minPoints = []
    maxPoints = []

    for searchIndex in range(searchCount):
        minSeed = None if randomState is None else randomState + 2 * searchIndex
        maxSeed = None if randomState is None else randomState + 2 * searchIndex + 1

        minPso = PSO(
            objFunc=lambda point: -modelFunc(point),
            bounds=bounds,
            popSize=popSize,
            iters=iters,
            randomState=minSeed,
        )
        bestMinPoint, _ = minPso.Optimize()
        minPoints.append(bestMinPoint)

        maxPso = PSO(
            objFunc=modelFunc,
            bounds=bounds,
            popSize=popSize,
            iters=iters,
            randomState=maxSeed,
        )
        bestMaxPoint, _ = maxPso.Optimize()
        maxPoints.append(bestMaxPoint)

    return np.array(minPoints), np.array(maxPoints)


# Compatibility aliases for legacy scripts.
pso_search_extrema = PsoSearchExtrema
