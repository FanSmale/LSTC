import numpy as np
import torch
from torch import nn

class TripleLabelAnn(nn.Module):

    def __init__(self, paraLayerNumNodes: list = None, paraActivators: str = "s" * 100, paraLearningRate: float = 0.05,
                 paraTrainDataMatrix: np.array = None, paraTrainLabelsMatrix: np.array = None,
                 paraTestDataMatrix: np.array = None, paraTestLabelsMatrix: np.array = None):
        '''
        Contruction, Create a neural network.

        :param paraLayerNumNodes: A list is used to describe each layer of network nodes.
        :param paraActivators: A string is used to describe each layer activators.
        :param paraLearningRate: Learning Rate.
        :param paraTrainDataMatrix: Input training data.
        :param paraTrainLabelsMatrix: Output training data.
        :param paraTestDataMatrix: Input testing data matrix for prediction.
        (If there is no testing set for prediction in the actual environment, part of the training set can be divided into a validation set).
        :param paraTestLabelsMatrix: Output testing data matrix for verifying prediction.
        '''

        super().__init__()

        # Initialization
        self.device = torch.device("cuda")
        self.trainDataMatrix = paraTrainDataMatrix
        self.trainLabelsMatrix = paraTrainLabelsMatrix
        self.testDataMatrix = paraTestDataMatrix
        self.testLabelsMatrix = paraTestLabelsMatrix

        # These parameters may be used outside the class
        # These are prediction of the network for the dataMatrix(Parameters of the getCurrentAccuary function)

        # The results have integrated all paired nodes, so an n * 3 matrix will be obtained
        self.tempPredictTensor = None
        # The results does not integrate all paired nodes, so an n * 6 matrix will be obtained
        self.tempPredictTensor6Port = None

        # Create a network
        tempModel = []
        for i in range(len(paraLayerNumNodes) - 1):
            tempInput = paraLayerNumNodes[i]
            tempOutput = paraLayerNumNodes[i + 1]
            tempLinear = nn.Linear(tempInput, tempOutput)
            tempModel.append(tempLinear)
            tempModel.append(getActivator(paraActivators[i]))
        self.model = nn.Sequential(*tempModel)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=paraLearningRate)
        self.lossFunction = nn.MSELoss()
        self.lossFunction = self.lossFunction.to(self.device)

    def forward(self, paraInput):

        return self.model(paraInput)

    def oneRoundTrain(self):
        '''
        Finish single training to trainingSet.

        :return: Loss value.
        '''

        # Output the predicted value
        tempInputTensor = torch.as_tensor(np.float32(self.trainDataMatrix)).to(self.device)
        tempOutputsTensor = self.model(tempInputTensor)
        # calculate the loss
        tempTargetsTensor = torch.as_tensor(np.float32(self.trainLabelsMatrix)).to(self.device)
        loss = self.lossFunction(tempOutputsTensor, tempTargetsTensor)
        # set the grad to zero
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def getCurrentAccuary(self, dataMatrix, labelMatrix):
        '''
        Return the predicted value

        :param dataMatrix: Data matrix that needs to be input into the network.
        :param labelMatrix: Label matrix for validation.
        :return: Predicted Accuracy.
        '''

        # Get the output (prediction) of the dataMatrix after passing through the network
        tempInputTensor = torch.as_tensor(np.float32(dataMatrix)).to(self.device)
        self.tempPredictTensor6Port = self.model(tempInputTensor)

        # Predict the accuracy of paired output nodes with a threshold of 0.5
        tempSwitch = self.tempPredictTensor6Port[:, ::2] < self.tempPredictTensor6Port[:, 1::2]
        self.tempPredictTensor = tempSwitch.int()
        tempPredictMatrix = self.tempPredictTensor.cpu().numpy()    # Subsequent operations require numpy

        tempSum = 0
        tempCorrect = 0

        for predictLine, targetLine in zip(tempPredictMatrix, labelMatrix):
            # It is only necessary to judge whether the prediction of the main label is accurate.
            # Auxiliary labels do not need to be predicted.
            if (predictLine[0] == targetLine[0]):
                tempCorrect += 1
            tempSum += 1
        return float(tempCorrect / tempSum)

    def boundedTrain(self, paraLowerRounds: int = 200, paraCheckingRounds: int = 200, paraEnhancementThreshold: float = 0.001):
        '''
        Multiple training on the data.

        :param paraLowerRounds: Rounds of Bounded train.
        :param paraCheckingRounds: Periodic output of current training round.
        :param paraEnhancementThreshold: The Precision of train.
        :return: Final testingSet predictive accuracy.
        '''
        print("***Bounded train***")
        for i in range(paraLowerRounds):
            if i % 100 == 0:
                print("round: ", i)
            self.oneRoundTrain()

        # Step 3. Train more rounds.
        # Continue training, and stop when the improvement is below a certain threshold
        print("***Precision train***")
        i = paraLowerRounds
        lastTrainingAccuracy = 0
        while True:
            tempLoss = self.oneRoundTrain()
            # Encountered checkpoint, output training effect
            if i % paraCheckingRounds == paraCheckingRounds - 1:
                # Judging the current training state by Accuracy
                tempAccuracy = self.getCurrentAccuary(self.testDataMatrix, self.testLabelsMatrix)
                print("Regular round: ", (i + 1), ", training accuracy = ", tempAccuracy)
                if lastTrainingAccuracy > tempAccuracy - paraEnhancementThreshold:
                    break  # No more enhancement.
                else:
                    lastTrainingAccuracy = tempAccuracy
                print("The loss is: ", tempLoss)
            i += 1

        result = self.getCurrentAccuary(self.testDataMatrix, self.testLabelsMatrix)
        print("Finally, the accuracy is ", result)
        return result

def getActivator(paraActivator: str = 's'):
    '''
    Parsing the specific char of activator.

    :param paraActivator: specific char of activator.
    :return: Activator layer.
    '''
    if paraActivator == 's':
        return nn.Sigmoid()
    elif paraActivator == 'r':
        return nn.ReLU()
    elif paraActivator == 'l':
        return nn.LeakyReLU()
    elif paraActivator == 'e':
        return nn.ELU()
    elif paraActivator == 'u':
        return nn.Softplus()
    elif paraActivator == 'o':
        return nn.Softsign()
    else:
        return nn.Sigmoid()


