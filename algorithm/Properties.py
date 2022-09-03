import os
import json


class Properties:
    """
    The algorithm parameters.
    """

    def __init__(self, datasetName: str = "default"):

        # check whether the JSON file exists
        configName = "../configuration/config.json"
        assert os.path.exists(configName), "Config file is not accessible."
        # open json
        configName = str(configName)
        with open(configName) as f:
            cfg = json.load(f)["lstc"]

        # about nervous network
        self.learningRate = cfg["common"]["learningRate"]
        self.enhancementThreshold = cfg["common"]["enhancementThreshold"]
        self.kFoldNum = cfg["common"]["crossNum"]

        # about train parameters
        self.boundedTrainRounds = cfg["common"]["boundedTrainRounds"]
        self.enhancementThreshold = cfg["common"]["enhancementThreshold"]

        # about dataset
        self.trainDataMatrix = cfg["common"]["trainDataMatrix"]
        self.trainLabelMatrix = cfg["common"]["trainLabelMatrix"]
        self.testDataMatrix = cfg["common"]["testDataMatrix"]
        self.testLabelMatrix = cfg["common"]["testLabelMatrix"]
        self.numInstances = cfg["common"]["numInstances"]
        self.numConditions = cfg["common"]["numConditions"]
        self.numLabels = cfg["common"]["numLabels"]

        # about feature conversion
        self.dc = cfg["common"]["dc"]
        self.controlRadio = cfg["common"]["controlRadio"]
        self.boundR = cfg["common"]["boundR"]

        # read dataset-specific parameter
        assert datasetName in cfg.keys(), "".join(["The parameters of ", datasetName, "are not defined in the JSON file of config."])
        tempDataSetCfg = cfg[datasetName]
        self.fileName = tempDataSetCfg["fileName"]
        self.activators = tempDataSetCfg["activators"]
        self.fullConnectLayerNumNodes = tempDataSetCfg["fullConnectLayerNumNodes"]
        self.outputFileSrcAndName = tempDataSetCfg["outputFileSrcAndName"]

        # read independent clustering radio
        if ("controlRadio" in tempDataSetCfg):
            self.controlRadio = tempDataSetCfg["controlRadio"]
        if ("boundR" in tempDataSetCfg):
            self.boundR = tempDataSetCfg["boundR"]
            # other parameters


    def initialization(self):

        # Additional data
        self.label2RepresentativeIndexMapping = dict()     # Index mapping of all representative instances corresponding to a single label (do not consider other auxiliary labels)
        self.instancesDistanceMatrix = None
        self.label2RepresentativeIndexArray = []                        # All representative instances corresponding to a label (consider other auxiliary labels)
        self.networkList = []
        self.predictLabelMatrix = None
        self.predictLabelMatrixDoublePort = None
        self.globalDensityArray = []
