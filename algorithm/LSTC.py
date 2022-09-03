#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/1
# @Author  : Xing-Yi Zhang
# @Email   : 1328365276@qq.com
# @File    : LSTC.py


from sklearn.model_selection import KFold
import numpy as np
import torch
import time
import scipy
import scipy.io as sio
import math
from algorithm.Properties import Properties
from algorithm.paired_output_network.TripleLabelAnn import TripleLabelAnn
from sklearn import metrics


class Lstc:
    '''
    LSTC: When label-specific features meet third-order label correlations
    '''

    def __init__(self, paraDataSet: Properties):
        '''
        Contruction: algorithm main process.

        :param paraDataSet: Related parameters about the algorithm.
        '''

        # Initialization
        timerStart = time.time()
        self.device = torch.device('cuda')
        self.manhattanDist = torch.nn.PairwiseDistance(p = 1, eps = 0, keepdim = True).to(self.device)
        self.dataset = paraDataSet
        self.peakF1Score = 0
        self.auc = 0
        self.NDCG = 0
        self.runTime = 0

        print('Global preprocessing...')
        # Global precompute: calculate the global density of each instance and obtain the instance adjacency matrix
        self.representativenessRankArray = self.computeGlocalDensityAndDistanceMatrix(self.dataset.trainDataMatrix, self.dataset.dc)
        # Pre-calculate the distance between training data instance and testing data instance, thereby preparing for the prediction
        self.testData2TrainDataDistanceMatrix = self.computeTestDataToCenterDistance(paraDataSet.trainDataMatrix,
                                                                                     paraDataSet.testDataMatrix)

        flag = 0
        controlRadio = self.dataset.controlRadio
        tempLabelPositiveNum = np.sum(self.dataset.trainLabelMatrix, axis=0)  # Count positive class in each label
        numLabel = paraDataSet.numLabels
        numInstance = self.dataset.numInstances

        # Train a network for each label
        for mainLabel in range(numLabel):
            print("*******************************| The progress:{0}/{1} |*******************************".format(mainLabel,numLabel))
            print('Feature Conversion for th-{0} label.'.format(mainLabel))

            # Find the most similar label and the least similar label
            mainLabel, mostSimilarLabel, leastSimilarLabel = self.constructRelevantLabel(mainLabel, self.dataset.trainLabelMatrix)
            print('Then the most similar label is th-{0} label.'.format(mostSimilarLabel))
            print('Then the least similar label is th-{0} label.\n'.format(leastSimilarLabel))

            # Initialize the array used to count the current number of representative index.
            representativeIndexCombinations = []

            # Get representative index for three labels
            for k in (mainLabel, mostSimilarLabel, leastSimilarLabel):

                # Avoid double calculate
                if k in self.dataset.label2RepresentativeIndexMapping:
                    print('{0} representative instances are obtained in the matrix '
                          'wrt label {1}-th Label.'.format(len(self.dataset.label2RepresentativeIndexMapping[k]), k))
                    representativeIndexCombinations += self.dataset.label2RepresentativeIndexMapping[k]
                    continue

                # Calculate the number of representative instance selections for matrices P and N
                positiveInstanceNum = tempLabelPositiveNum[k]
                negativeInstanceNum = numInstance - positiveInstanceNum
                selectNum = int(np.ceil(min(positiveInstanceNum * controlRadio, negativeInstanceNum * controlRadio)))

                tempIndexArray = []
                if (selectNum == 0):
                    tempIndexArray = []
                else:
                    selectNum = min(100, selectNum)         # Avoid too many representative instances
                    tempIndexArray = self.getRepresentativeInstanceIndex(dataMatrix = self.dataset.trainDataMatrix,
                                                                        labelMatrix = self.dataset.trainLabelMatrix,
                                                                        paraLabelIndex = k,
                                                                        paraSelectNum = selectNum).tolist()
                self.dataset.label2RepresentativeIndexMapping.update({k: tempIndexArray})
                representativeIndexCombinations += tempIndexArray
                print('{0} representative instances are obtained in the matrix '
                      'wrt label {1}-th Label (has been calculated).'.format(len(tempIndexArray), k))

            print('In the end, the number of all representative '
                  'instance is {}.\n'.format(len(representativeIndexCombinations)))

            self.dataset.label2RepresentativeIndexArray.append(representativeIndexCombinations)

            print("Construct the new training data matrix...")
            tempNewTrainDataMatrix, tempNewTrainLabelMatrix = self.constructNewDataMatrix(mainLabel, mostSimilarLabel, leastSimilarLabel)

            print(">> New train data matrix: {0}x{1}".format(tempNewTrainDataMatrix.shape[0],
                                                             tempNewTrainDataMatrix.shape[1]))
            print(">> New train label matrix: {0}x{1}".format(tempNewTrainDataMatrix.shape[0],
                                                              tempNewTrainDataMatrix.shape[1]))

            print("Construct the new testing data matrix (for testing this network)...")
            tempNewTestDataMatrix, tempNewTestLabelMatrix = self.constructNewDataMatrixForTesting(mainLabel)

            print(">> New train data matrix: {0}x{1}".format(tempNewTestDataMatrix.shape[0],
                                                             tempNewTestDataMatrix.shape[1]))
            print(">> New train label matrix: {0}x{1}\n".format(tempNewTestDataMatrix.shape[0],
                                                              tempNewTestDataMatrix.shape[1]))

            print("Begin to train the network for {0}-th label...".format(mainLabel))

            # Complementary input layer
            tempFullConnectLayerNumNodes = self.dataset.fullConnectLayerNumNodes.copy()

            tempFullConnectLayerNumNodes[0] = tempNewTrainDataMatrix.shape[1]
            tempFullConnectLayerNumNodes.append(tempNewTrainLabelMatrix.shape[1])

            localNetwork = TripleLabelAnn(paraLayerNumNodes = tempFullConnectLayerNumNodes,
                                          paraActivators = self.dataset.activators,
                                          paraLearningRate = self.dataset.learningRate,
                                          paraTrainDataMatrix = tempNewTrainDataMatrix,
                                          paraTrainLabelsMatrix = tempNewTrainLabelMatrix,
                                          paraTestDataMatrix = tempNewTestDataMatrix,
                                          paraTestLabelsMatrix = tempNewTestLabelMatrix).cuda()

            localNetwork.boundedTrain(paraLowerRounds = self.dataset.boundedTrainRounds,
                                      paraCheckingRounds = 100,
                                      paraEnhancementThreshold = self.dataset.enhancementThreshold)

            # Prediction
            # Collect the predicted output label vector of the network and combine these labels into a prediction matrix

            if (flag == 0):
                flag = 1
                # Get single predicted label
                self.dataset.predictLabelMatrix = localNetwork.tempPredictTensor[:,0: 1]
                # Get double predicted label (to achieve Peak F1-score evaluation measure)
                self.dataset.predictLabelMatrixDoublePort = localNetwork.tempPredictTensor6Port[:,0: 2]
            else:
                self.dataset.predictLabelMatrix = torch.cat(
                    [self.dataset.predictLabelMatrix, localNetwork.tempPredictTensor[:, 0:1]], 1)
                self.dataset.predictLabelMatrixDoublePort = torch.cat(
                    [self.dataset.predictLabelMatrixDoublePort, localNetwork.tempPredictTensor6Port[:, 0:2]], 1)

        timerEnd = time.time()
        self.runTime = (timerEnd - timerStart)
        print("Compute the AUC...")
        self.auc = self.computeAUC()
        print("Compute the Peak F1-score ...")
        self.peakF1Score = self.computepeakF1Score()
        print("Compute the NDCG...")
        self.NDCG = self.computeNDCG()

    def constructRelevantLabel(self, paraMainLabelIndex, paraLabelMatrix):
        '''
        Find the most similar label and the least similar label based on main label.

        :param paraMainLabel: The index of main label
        :param paraLabelMatrix: Label matrix
        :return:
        '''

        labelMatrix = torch.from_numpy(paraLabelMatrix.transpose())
        labelVector = labelMatrix[paraMainLabelIndex]

        maxIndex = -1
        maxDistance = -1
        minIndex = -1
        minDistance = float('inf')

        for tempLabelIndex in range(self.dataset.numLabels):
            if (tempLabelIndex == paraMainLabelIndex):
                continue
            tempDistance = self.manhattanDist(labelVector, labelMatrix[tempLabelIndex])
            if tempDistance > maxDistance:
                maxDistance = tempDistance
                maxIndex = tempLabelIndex
            if tempDistance < minDistance:
                minDistance = tempDistance
                minIndex = tempLabelIndex
        return paraMainLabelIndex, maxIndex, minIndex

    def computepeakF1Score(self):
        '''
        Compute the Peak F1-score

        :return: The Peak F1-score
        '''
        tempPredictMatrix = self.dataset.predictLabelMatrixDoublePort
        tempTargetMatrix = self.dataset.testLabelMatrix

        tempProbaMatrix = torch.exp(tempPredictMatrix[:, 1::2]) / \
                          (torch.exp(tempPredictMatrix[:, 1::2]) + torch.exp(tempPredictMatrix[:, ::2]))
        tempProbVector = tempProbaMatrix.reshape(-1).cpu().detach().numpy()
        temp = np.argsort(-tempProbVector)
        tempTargetVector = tempTargetMatrix.reshape(-1)
        allLabelSort = tempTargetVector[temp]

        tempYF1 = np.zeros(temp.size)

        allTP = np.sum(tempTargetVector == 1)

        for i in range(temp.size):

            TP = np.sum(allLabelSort[0:i + 1] == 1)
            P = TP / (i + 1)
            R = TP / allTP
            if (P + R) == 0:
                tempYF1[i] = 0
            else:
                tempYF1[i] = 2.0 * P * R / (P + R)
        return np.max(tempYF1)

    def computeAUC(self):
        '''
        Compute the AUC

        :return: The AUC
        '''
        tempPredictMatrix = self.dataset.predictLabelMatrixDoublePort
        tempTargetMatrix = self.dataset.testLabelMatrix

        tempProbaMatrix = torch.exp(tempPredictMatrix[:, 1::2]) / \
                          (torch.exp(tempPredictMatrix[:, 1::2]) + torch.exp(tempPredictMatrix[:, ::2]))
        tempProbVector = tempProbaMatrix.reshape(-1).cpu().detach().numpy()
        tempTargetVector = tempTargetMatrix.reshape(-1)
        return metrics.roc_auc_score(tempTargetVector, tempProbVector)

    def computeNDCG(self):
        '''
        Compute the NDCG

        :return: the NDCG
        '''
        tempPredictMatrix = self.dataset.predictLabelMatrixDoublePort
        tempTargetMatrix = self.dataset.testLabelMatrix

        tempProbaMatrix = torch.exp(tempPredictMatrix[:, 1::2]) / \
                          (torch.exp(tempPredictMatrix[:, 1::2]) + torch.exp(tempPredictMatrix[:, ::2]))
        # 获得概率序列与原目标序列
        tempProbVector = tempProbaMatrix.reshape(-1).cpu().detach().numpy()
        tempTargetVector = tempTargetMatrix.reshape(-1)

        # 按照概率序列排序原1/0串
        temp = np.argsort(-tempProbVector)
        allLabelSort = tempTargetVector[temp]

        # 获得最佳序列: 1111...10000...0
        sortedTargetVector = np.sort(tempTargetVector)[::-1]

        # compute DCG(使用预测的顺序, rel是真实顺序, 实际是111110111101110000001000100
        DCG = 0
        for i in range(temp.size):
            rel = allLabelSort[i]
            denominator = np.log2(i + 2)
            DCG += (rel / denominator)

        # compute iDCG(使用最佳顺序: 11111111110000000000)
        iDCG = 0
        for i in range(temp.size):
            rel = sortedTargetVector[i]
            denominator = np.log2(i + 2)
            iDCG += (rel / denominator)

        return DCG / iDCG

    def getRepresentativeInstanceIndex(self, dataMatrix, labelMatrix, paraLabelIndex, paraSelectNum):
        '''
        Divide the data matrix based on a certain label,
        then obtain the index of the same number of representative instances in each divided matrix respectively.

        :param dataMatrix: Data matrix (X)
        :param labelMatrix: Label matrix (Y)
        :param paraLabelIndex: Benchmark label (k)
        :param paraSelectNum: The number of representative instances to be selected in each matrix after division (m_k)
        :return:
        '''

        # divide X^{(k)} into P^{(k)} and N^{(k)}
        indexOfPositive = labelMatrix[:, paraLabelIndex] == 1
        indexOfNegative = labelMatrix[:, paraLabelIndex] == 0
        numberOfPositive = sum(indexOfPositive)
        numberOfNegative = sum(indexOfNegative)

        if numberOfPositive >= paraSelectNum:
            positiveCenter = self.computeInstanceRepresentativenessForPositive(dataMatrix, indexOfPositive, self.dataset.dc, paraSelectNum)
        else:
            positiveCenter = dataMatrix[indexOfPositive, :]

        if numberOfNegative >= paraSelectNum:
            negativeCenter = self.computeInstanceRepresentativenessForNegative(dataMatrix, indexOfNegative, indexOfPositive, self.dataset.dc, paraSelectNum)
        else:
            negativeCenter = dataMatrix[indexOfNegative, :]

        return np.hstack([positiveCenter, negativeCenter])

    def computeInstanceRepresentativenessForPositive(self, dataMatrix, paraIndexOfPositive, paraDc, paraSelectNum):
        '''
        Select a specified number of representative instances from the positive instance matrix.

        :param dataMatrix: Data matrix.
        :param paraIndexOfPositive: A bool array reflecting the position of the local index in the global index.
        :param paraDc: A parameter about density domain.
        :param paraSelectNum: The selection number of representative instance.
        :return: The index of representative instance in global data matrix.
        '''


        # Get the mapping from local index to global index
        tempNaturalOrder = np.array(range(dataMatrix.shape[0]))
        localIndex2GlobalIndex = tempNaturalOrder[paraIndexOfPositive]
        localPositiveInstanceNum = len(localIndex2GlobalIndex)

        # Initialize local density array (it uses local index and stores the local density value)
        dividingDensityArray = []

        tempDcSquare = 1.0 * paraDc * paraDc

        for i in localIndex2GlobalIndex:        # Here i is a global index
            rho_i = 0
            for j in localIndex2GlobalIndex:    # j is a global index as well
                if i != j:
                    d_ij = self.dataset.instancesDistanceMatrix[i][j]
                    rho_i += math.exp(np.multiply(-d_ij, d_ij) / tempDcSquare)
            dividingDensityArray.append(rho_i)

        dividingDensityArray = np.array(dividingDensityArray)
        tempDistanceToMasterArray = np.zeros(localPositiveInstanceNum)

        for i in range(localPositiveInstanceNum):# Here i is a local index
            # tempIndexArray store the local indices
            tempIndexArray = np.argwhere(dividingDensityArray > dividingDensityArray[i])
            if (tempIndexArray.size > 0):
                globalIndex_i = localIndex2GlobalIndex[i]
                globalIndexArray = localIndex2GlobalIndex[tempIndexArray]
                tempDistanceToMasterArray[i] = np.min(self.dataset.instancesDistanceMatrix[globalIndex_i, globalIndexArray])

        # Representativeness
        representativenessArray = dividingDensityArray * tempDistanceToMasterArray

        # Select the most similar instances through sort
        representativenessIndexArray = []
        for i in range(paraSelectNum):
            maxRepresentativenessIndex = i
            for j in range(i + 1, len(representativenessArray)):
                if representativenessArray[j] > representativenessArray[maxRepresentativenessIndex]:
                    maxRepresentativenessIndex = j
            representativenessArray[maxRepresentativenessIndex], representativenessArray[i] \
                = representativenessArray[i], representativenessArray[maxRepresentativenessIndex]
            representativenessIndexArray.append(maxRepresentativenessIndex)

        # convert local index to global index
        return localIndex2GlobalIndex[representativenessIndexArray]

    def computeInstanceRepresentativenessForNegative(self, dataMatrix, paraIndexOfNegative, paraIndexOfPositive, paraDc, paraSelectNum):
        '''
        Select a specified number of representative instances from the negative instance matrix.

        :param dataMatrix: Data matrix.
        :param paraIndexOfNegative: A bool array reflecting the negative of the local index in the global index.
        :param paraDc: A parameter about density domain.
        :param paraSelectNum: The selection number of representative instance.
        :return: The index of representative instance in global data matrix.
        '''

        # Get the mapping from local index to global index
        tempNaturalOrder = np.array(range(dataMatrix.shape[0]))
        negativeLocalIndex2GlobalIndex = tempNaturalOrder[paraIndexOfNegative]
        positiveLocalIndex2GlobalIndex = tempNaturalOrder[paraIndexOfPositive]

        # Initialize local density array (it uses local index and stores the local density value)
        dividingDensityArray = []

        tempDcSquare = 1.0 * paraDc * paraDc
        globalDensityArray = self.dataset.globalDensityArray

        for i in negativeLocalIndex2GlobalIndex:               # Here i is a global index
            rho_i = globalDensityArray[i]
            for j in positiveLocalIndex2GlobalIndex:           # j is a global index as well
                d_ij = self.dataset.instancesDistanceMatrix[i][j]
                rho_i - math.exp(np.multiply(-d_ij, d_ij) / tempDcSquare)
            dividingDensityArray.append(rho_i)

        tempDistanceToMasterArray = np.zeros(len(dividingDensityArray))
        dividingDensityArray = np.array(dividingDensityArray)

        for i in range(len(dividingDensityArray)):
            tempIndex = np.argwhere(dividingDensityArray > dividingDensityArray[i])
            if (tempIndex.size > 0):
                globalIndex_i = negativeLocalIndex2GlobalIndex[i]
                globalIndexArray = negativeLocalIndex2GlobalIndex[tempIndex]
                tempDistanceToMasterArray[i] = np.min(self.dataset.instancesDistanceMatrix[globalIndex_i, globalIndexArray])

        # Representativeness
        representativenessArray = dividingDensityArray * tempDistanceToMasterArray

        # Select the most similar instances through sort
        representativenessIndexArray = []
        for i in range(paraSelectNum):
            maxRepresentativenessIndex = i
            for j in range(i + 1, len(representativenessArray)):
                if representativenessArray[j] > representativenessArray[maxRepresentativenessIndex]:
                    maxRepresentativenessIndex = j
            representativenessArray[maxRepresentativenessIndex], representativenessArray[i] = representativenessArray[i], representativenessArray[maxRepresentativenessIndex]
            representativenessIndexArray.append(maxRepresentativenessIndex)

        # convert local index to global index
        return negativeLocalIndex2GlobalIndex[representativenessIndexArray]

    def constructNewDataMatrix(self, paraMainLabel, paraMostSimilarLabel, paraLeastSimilarLabel):
        '''
        Construct a new data matrix through its instances and selected representative instances.

        :param paraMainLabel: The benchmark label.
        :param paraMostSimilarLabel: the most similar label.
        :param paraLeastSimilarLabel: the least similar label.
        :return: A new matrix (n * 2m_k) and a new label matrix of three labels.
        '''

        tempNewDataMatrix = []
        tempNewLabelMatrix = []
        tempRepresentativeIndexArray = np.array(self.dataset.label2RepresentativeIndexArray[paraMainLabel])

        for tempInstanceIndex in range(len(self.dataset.trainDataMatrix)):
            tempXLine = []
            tempYLine = []

            # Constructs the new label information of the current line
            for tempLabel in (paraMainLabel, paraMostSimilarLabel, paraLeastSimilarLabel):
                tempYLine += ([0, 1] if self.dataset.trainLabelMatrix[tempInstanceIndex][tempLabel] == 1 else [1, 0])
            tempNewLabelMatrix.append(tempYLine)

            # Constructs the new instance information of the current line
            for tempRepresentativeIndex in tempRepresentativeIndexArray:
                tempXLine.append(self.dataset.instancesDistanceMatrix[tempInstanceIndex][tempRepresentativeIndex])
            tempNewDataMatrix.append(tempXLine)

        return np.array(tempNewDataMatrix), np.array(tempNewLabelMatrix)

    def constructNewDataMatrixForTesting(self, paraMainLabel):
        '''
        Construct a new data testing matrix through its instances and selected representative instances that has been trained.

        :param paraMainLabel: The benchmark label (No auxiliary label information is required for prediction).
        :return: A new matrix (n * 2m_k) and a new label matrix of one labels.
        '''

        tempNewDataMatrix = []
        tempNewLabelMatrix = []
        tempRepresentativeIndexArray = np.array(self.dataset.label2RepresentativeIndexArray[paraMainLabel])

        # Constructs the new label information of the current line
        for tempInstanceIndex in range(len(self.dataset.testDataMatrix)):
            tempXLine = []

            # Constructs the new label information of the current line
            tempNewLabelMatrix.append(self.dataset.testLabelMatrix[tempInstanceIndex][paraMainLabel])

            # Constructs the new instance information of the current line
            for tempRepresentativeIndex in tempRepresentativeIndexArray:
                tempXLine.append(self.testData2TrainDataDistanceMatrix[tempInstanceIndex][tempRepresentativeIndex])

            tempNewDataMatrix.append(tempXLine)

        return np.array(tempNewDataMatrix), np.array(tempNewLabelMatrix).reshape(-1, 1)

    def computeGlocalDensityAndDistanceMatrix(self, dataMatrix, paraDc):
        '''
        Calculate the global density of each instance and obtain the instance adjacency matrix.

        :param self: The glocal dataMatrix that need to train.
        :param paraDc: The dc ratio.
        '''

        # Generate a array about the distance between two node, the array length is n (n - 1) / 2
        tempDist = torch.nn.functional.pdist(torch.from_numpy(dataMatrix)).to(self.device)

        # Convert to an n * n matrix
        tempDistancesMatrix = scipy.spatial.distance.squareform(tempDist.cpu().numpy(), force='no', checks=True)

        # Density calculation under Gaussian kernel
        tempDcSquare = 1.0 * paraDc * paraDc

        # Get global density array
        self.dataset.globalDensityArray = np.sum(np.exp(np.multiply(-tempDistancesMatrix, tempDistancesMatrix) / tempDcSquare), axis = 0)

        # Get instance adjacency matrix
        self.dataset.instancesDistanceMatrix = tempDistancesMatrix

    def computeTestDataToCenterDistance(self, trainDataMatrix, testDataMatrix):
        '''
        Construct the distance adjacency matrix between the training data matrix and testing data matrix.
        This adjacency matrix is used to save the time, when we need calculate the distance between the
        tesing instance and the representative instance of the training matrix.

        :param trainDataMatrix: Train data matrix.
        :param testDataMatrix:  Test data matrix.
        :return: This adjacency matrix between training instance and testing instance.
        '''

        tempMatrix = np.vstack([trainDataMatrix, testDataMatrix])
        tempDist = torch.nn.functional.pdist(torch.from_numpy(tempMatrix).float()).to(self.device)
        # Convert to an n * n matrix
        combinatorialMatrix = scipy.spatial.distance.squareform(tempDist.cpu().numpy(), force='no', checks=True)

        return combinatorialMatrix[trainDataMatrix.shape[0]:, 0: trainDataMatrix.shape[0]]

def fullTrain(paraDataSetName: str = 'Emotion', isCrossValidation: bool = False):
    '''
    Common training

    :param paraDataSetName: The name of dataset
    :param isCrossValidation: Whether to use cross-validation
    '''

    print("The dataset is: ", paraDataSetName)
    prop = Properties(paraDataSetName)

    if not isCrossValidation:
        prop.initialization()
        prop.trainDataMatrix, prop.trainLabelMatrix, prop.testDataMatrix, prop.testLabelMatrix, prop.numInstances, prop.numConditions, prop.numLabels = matReader(
            prop.fileName)
        tempLstc = Lstc(prop)
        print("AUC: ", tempLstc.auc)
        print("Peak F1-Score: ", tempLstc.peakF1Score)
        print("NDCG: ", tempLstc.NDCG)
        print('Runtime: ', tempLstc.runTime, "s")
    else:
        kf = KFold(prop.kFoldNum, shuffle=True)
        tempTrainDataMatrix, tempTrainLabelMatrix, tempTestDataMatrix, tempTestLabelMatrix, _, _, _ = matReader(prop.fileName)

        tempDataMatrix = np.vstack((tempTrainDataMatrix, tempTestDataMatrix))
        tempLabelMatrix = np.vstack((tempTrainLabelMatrix, tempTestLabelMatrix))

        aucRecord = np.zeros((prop.kFoldNum, 1))
        peakF1ScoreRecord = np.zeros((prop.kFoldNum, 1))
        NDCGRecord = np.zeros((prop.kFoldNum, 1))
        runtimeRecord = np.zeros((prop.kFoldNum, 1))

        for k, (trainIndex, testIndex) in enumerate(kf.split(tempDataMatrix)):
            # refresh some parameters
            prop.initialization()

            # confirm data set
            prop.trainDataMatrix = tempDataMatrix[trainIndex, :]
            prop.testDataMatrix = tempDataMatrix[testIndex, :]
            # confirm label set
            prop.trainLabelMatrix = tempLabelMatrix[trainIndex, :]
            prop.testLabelMatrix = tempLabelMatrix[testIndex, :]
            # confirm other information
            prop.numInstances = prop.trainDataMatrix.shape[0]
            prop.numConditions =prop.trainDataMatrix.shape[1]
            prop.numLabels = prop.trainLabelMatrix.shape[1]

            #sys.stdout = open(os.devnull, 'w') # close print
            tempLstc = Lstc(prop)
            #sys.stdout = sys.__stdout__        # open print

            peakF1ScoreRecord[k,0] = tempLstc.peakF1Score
            NDCGRecord[k,0] = tempLstc.NDCG
            aucRecord[k, 0] = tempLstc.auc
            runtimeRecord[k, 0] = tempLstc.runTime

            print("AUC: ", tempLstc.auc)
            print("Peak F1-Score: ", tempLstc.peakF1Score)
            print("NDCG: ", tempLstc.NDCG)
            print('Runtime: ', tempLstc.runTime, "s")

        print("******************* {0}-Fold Cross Validation *****************".format(prop.kFoldNum) )
        print("AUC: ")
        print(aucRecord)
        print("The mean and variance of AUC: ""{0}±{1}".format(
            round(aucRecord.mean(),4), round(aucRecord.std(),4)))
        print("Peak F1-Score: ")
        print(peakF1ScoreRecord)
        print("The mean and variance of peakF1Score: ""{0}±{1}".format(
            round(peakF1ScoreRecord.mean(), 4), round(peakF1ScoreRecord.std(), 4)))
        print("NDCG: ")
        print(NDCGRecord)
        print("The mean and variance of NDCG: {0}±{1}".format(
            round(NDCGRecord.mean(),4), round(NDCGRecord.std(),4)))
        print('runtime: ')
        print(runtimeRecord)
        print("The mean and variance of runtime: {0}±{1}".format(
            round(runtimeRecord.mean(),4), round(runtimeRecord.std(),4)))

        # save to local
        outputContent = paraDataSetName
        outputContent += "\nAUC is {0}±{1}".format(
            round(aucRecord.mean(),4), round(aucRecord.std(),4))
        outputContent += "\nPeak F1-score is {0}±{1}".format(
            round(peakF1ScoreRecord.mean(), 4), round(peakF1ScoreRecord.std(), 4))
        outputContent += "\nNDCG is {0}±{1}".format(
            round(NDCGRecord.mean(),4), round(NDCGRecord.std(),4))
        outputContent += "\nRuntime is {0}±{1}".format(
            round(runtimeRecord.mean(),4), round(runtimeRecord.std(),4))

        with open(prop.outputFileSrcAndName, "w")as f:
            f.write(outputContent)
            f.close()

def matReader(paraDataSetName: str = "../datasets/DataMat/Emotion.mat"):
    '''
    read the dataset file (.mat).

    :param paraDataSetName: Dataset file (.mat) path
    :return: Normalized training data matrix, training label matrix, normalized testing data matrix, testing label matrix,
    the number of training data matrix, the feature number of data matrix, the label number of label matrix.
    '''

    tempDataSet = sio.loadmat(paraDataSetName)

    # get train data and labels
    tempTrainData = np.array(tempDataSet['train_data'])
    tempTrainLabels = np.array(tempDataSet['train_target']).transpose()

    # get test data and labels
    tempTestData = np.array(tempDataSet['test_data'])
    tempTestLabels = np.array(tempDataSet['test_target']).transpose()

    # normalization
    tempTrainDataNorm = (tempTrainData - tempTrainData.min(axis=0)) / \
                  (tempTrainData.max(axis=0) - tempTrainData.min(axis=0) + 0.0001)
    tempTestDataNorm = (tempTestData - tempTestData.min(axis=0)) / \
                 (tempTestData.max(axis=0) - tempTestData.min(axis=0) + 0.0001)

    # convert -1 to 0
    tempTrainLabels[tempTrainLabels == -1] = 0
    tempTestLabels[tempTestLabels == -1] = 0

    return tempTrainDataNorm, tempTrainLabels, tempTestDataNorm, tempTestLabels, \
           tempTrainDataNorm.shape[0], tempTrainDataNorm.shape[1], tempTrainLabels.shape[1]


if __name__ == '__main__':

    # 17 datasets, choose one for experimentation.
    # Some of them are datasets of non-text domain:
    # "Birds", "Cal500", "CHD49", "Emotion", "Flags", "Foodtruck", "Image", "GpositiveGO", "Scene", "VirusGO", "WaterQuality", "Yeast"
    # And the other part is a dataset of text domain:
    # "Art", "Business", "Enron", "Recreation", "Social"

    fullTrain(paraDataSetName = "Flags", isCrossValidation = False)


