import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from math import sqrt


class RecoPearson:
    trainNan = None
    train0 = None
    trainBool = None
    trainMean = None
    test = None
    reco = None

    def getCSVFiles(self, trainPath, testPath):
        train = pd.read_csv(trainPath, header=None, names=['moviesId', 'userId', 'Ratings'], encoding='cp437')
        self.test = pd.read_csv(testPath, header=None, names=['moviesId', 'userId', 'Ratings'], encoding='cp437')
        return train

    def fitPivotTable(self, trainPath, testPath):
        print(trainPath, testPath)
        train = self.getCSVFiles(trainPath, testPath)
        print('fit Nan tables')
        self.trainNan = pd.pivot_table(train, index='userId', columns='moviesId', aggfunc=np.max).astype('float32')
        self.trainNan.columns = [j for i, j in self.trainNan.columns]
        self.trainNan.reset_index()
        # used for matmul operations since with Nan output is all Nan
        print('fit Num tables')
        self.train0 = pd.pivot_table(train, index='userId', columns='moviesId', aggfunc=np.max,
                                     fill_value=float(0)).astype('float32')
        self.train0.columns = self.trainNan.columns
        self.train0.reset_index()
        return self.trainNan, self.train0

    def toboolean(self):
        # Works for Nan values as well, replaces with zero
        self.trainBool = (self.trainNan > 0).astype('float32')
        return self.trainBool

    def getRowMean(self):
        print('Get mean of each Row')
        return np.nan_to_num(np.nanmean(self.trainNan, axis=1).reshape((len(self.trainNan), 1)))

    def getMeanMatrix(self):
        print('Get Mean Matrix')
        self.trainMean = self.getRowMean()
        return self.trainMean * self.toboolean()

    def getRelRate(self):
        print('Get Relative Rate')
        return self.train0 - self.getMeanMatrix()

    def getCovarMatrixNumerator(self, ARel):
        print('Get Covariance matrix num')
        return np.dot(self.train0, self.train0.T)

    def getNormalizedRatingEachUser(self):
        ASquared = self.train0 * self.train0
        print(ASquared)
        print(f'ASquared shape is: {ASquared.shape}')
        ASumSquared = np.nansum(ASquared, axis=1).reshape((len(self.train0), 1))
        print(ASumSquared)
        ASquared = False
        print(f'ASumSquared shape is: {ASumSquared.shape}')
        return np.sqrt(ASumSquared)

    def getCovarMatrixDenominator(self, ARel):
        ASqrt = self.getNormalizedRatingEachUser()
        return ASqrt * ASqrt.T

    def getCovarMatrix(self, ARel):
        print('Get Weight matrix numerator term')
        WNum = self.getCovarMatrixNumerator(ARel)
        print('Get Weight matrix denominator term')
        WDenom = self.getCovarMatrixDenominator(ARel)
        return np.divide(WNum, WDenom, out=np.zeros_like(WNum), where=WDenom != 0)

    def getWMeanMatrix(self, Weights):
        return np.dot(Weights, self.trainBool)

    def getDelta(self, Weights, ARel):
        deltaNum = np.dot(Weights, ARel)
        deltaDenom = self.getWMeanMatrix(Weights)
        return np.divide(deltaNum, deltaDenom, out=np.zeros_like(deltaNum), where=deltaDenom != 0)

    def calcRec(self):
        ARel = self.getRelRate()
        Weights = self.getCovarMatrix(ARel)
        Delta = self.getDelta(Weights, ARel)
        self.reco = pd.DataFrame(self.trainMean + Delta, columns=list(self.train0.columns), index=list(self.train0.index))
        return self.reco

    def getRecofromModel(self, row):
        return self.reco.loc[row.userId, row.moviesId]

    def getRecoforTest(self):
        self.test['reco'] = self.test.apply(lambda row: self.getRecofromModel(row), axis=1)

    def getError(self):
        print(f'The Root mean square error is: {sqrt(mse(self.test.Ratings, self.test.reco))}')
        print(f'The Maximum Absolute error is: {mae(self.test.Ratings, self.test.reco)}')


if __name__ == "__main__":
    trainPath = sys.argv[1]
    testPath = sys.argv[2]

    inst = RecoPearson()

    inst.fitPivotTable(trainPath, testPath)
    print(inst.train0.head())

    Rec = inst.calcRec()
    print(Rec.head())

    inst.getRecoforTest()
    print(inst.test.head())

    inst.getError()

