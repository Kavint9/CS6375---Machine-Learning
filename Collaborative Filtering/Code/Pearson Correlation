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
        print(f'Getting training data from: {trainPath}')
        train = pd.read_csv(trainPath, header=None, names=['moviesId', 'userId', 'Ratings'], encoding='cp437')
        print(f'Getting testing data from: {testPath}')
        self.test = pd.read_csv(testPath, header=None, names=['moviesId', 'userId', 'Ratings'], encoding='cp437')
        return train

    def fitPivotTable(self, trainPath, testPath):
        train = self.getCSVFiles(trainPath, testPath)
        print('Fit training data into Pivot Table')

        # train Nan is used to calculate mean per userId using nanmean
        self.trainNan = pd.pivot_table(train, index='userId', columns='moviesId', aggfunc=np.max).astype('float32')

        # Pivot table columns are in tuples (Rating, movieId) reindex to just moviesId
        self.trainNan.columns = [j for i, j in self.trainNan.columns]
        self.trainNan.reset_index()

        # train0 is pivot table with 0 instead of nan for matrix multiplication operations
        # nan_to_num was slower than this approach

        self.train0 = pd.pivot_table(train, index='userId', columns='moviesId', aggfunc=np.max,
                                     fill_value=float(0)).astype('float32')
        self.train0.columns = self.trainNan.columns
        self.train0.reset_index()
        return self.trainNan, self.train0

    def toboolean(self):
        # replace every entry in the pivot table with 1
        # multiplying with this table gives only those values with a rating in the training set
        # Works for Nan values as well, replaces with zero
        self.trainBool = (self.trainNan > 0).astype('float32')
        return self.trainBool

    def getRowMean(self):
        print('Get mean of each Row')
        # Mean of each row gets calculated as a series and needs to be reshaped to a column vector
        return np.nan_to_num(np.nanmean(self.trainNan, axis=1).reshape((len(self.trainNan), 1)))

    def getMeanMatrix(self):
        print('Get each user\'s average rating')
        self.trainMean = self.getRowMean()
        # Return a matrix of the same dimensions as the training pivot table but with
        return self.trainMean * self.toboolean()

    def getRelRate(self):
        print('Compute matrix for each rating - avg. rating of user')
        return self.train0 - self.getMeanMatrix()

    def getCovarMatrixNumerator(self, ARel):
        print('Compute Numerator of Covariance Matrix')
        return np.dot(ARel, ARel.T)

    def getCovarMatrixDenominator(self, ARel):
        print('Compute square of relative ratings for each user ')
        asquare = ARel * ARel
        self.train0 = False
        adenomsquare = np.dot(asquare, self.trainBool.T) * np.dot(self.trainBool, asquare.T)
        return np.sqrt(adenomsquare)

    def getCovarMatrix(self, ARel):
        print('Get Weight matrix numerator term')
        wNum = self.getCovarMatrixNumerator(ARel)
        print('Get Weight matrix denominator term')
        wDenom = self.getCovarMatrixDenominator(ARel)
        return np.divide(wNum, wDenom, out=np.zeros_like(wNum), where=wDenom != 0)

    def getWMeanMatrix(self, Weights):
        return np.dot(Weights, self.trainBool)

    def getDelta(self, Weights, ARel):
        deltaNum = np.dot(Weights, ARel)
        deltaDenom = self.getWMeanMatrix(Weights)
        return np.divide(deltaNum, deltaDenom, out=np.zeros_like(deltaNum), where=deltaDenom != 0)


    def calcRec(self):
        aRel = self.getRelRate()
        weights = self.getCovarMatrix(aRel)
        delta = self.getDelta(weights, aRel)
        self.reco = pd.DataFrame(self.trainMean + delta, columns=list(self.trainNan.columns), index=list(self.trainNan.index))
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

