# LSTC
LSTC (LSTC: When label-specific features meet third-order label correlations) Algorithm

Abstract:
Multi-label learning (MLL) copes with applications such as text classification and image recognition where each instance is associated with multiple labels. Currently, some label specific feature extraction approaches such as LIFT serve each individual label well, however do not take advantage label correlation. Some label correlation exploiting approaches such as GLOCAL have independent model prediction and label correlation stages, therefore limit their learning ability. This paper introduces LSTC, a new algorithm that exploits label-specific features and third-order label correlations. On the one hand, for each label, we choose the same number of positive and negative representative instances inspired by density peaks. These instances are used to generate a new data matrix with label-specific features. On the other hand, we train a paired output prediction network for each label based on these matrices of its own and that of two auxiliary labels. In this way, we are essentially considering third-order label correlations. Specifically, the two auxiliary labels are the most similar and the least similar, respectively, to avoid getting stuck in local optima. Experiments are conducted on twelve benchmark datasets in comparison with eight popular algorithms. Results on three ranking-based measures demonstrate the overwhelming superiority of LSTC on data from various domains except the text-domain.

Keyword:
Label correlations, Label-specific features, Multi-label learning, Neural network.

Highlights:
We adopt density peaks under Gaussian kernel to extract label-specific features.
We train the model for each label using the data of two auxiliary labels.
We use the softmax function to obtain numerical predictions from paired outputs.
Our algorithm shows superiority in ranking-based measures of non-text data.

python 3.6
