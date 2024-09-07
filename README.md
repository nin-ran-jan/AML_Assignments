# AML_Assignments
Assignments done in the Advanced Machine Learning course at IIIT-Delhi. 

We were introduced to a variety of topics. Each of the assignments covered the following topics:

1. Noise Mitigation techniques.
2. Continual Learning techniques. 
3. Positive Unlabelled (PU) Classification techniques.

Each assignment folder has a report for further details on the experiements. 



# Assignment 2
The methodology employed in this assignment is test-time adaptation combined with an autoencoder-classifier architecture for handling sequential or time-series data. The key steps are:

Autoencoder-Based Feature Learning:

The model uses an autoencoder to compress input features into a lower-dimensional space. The encoder learns useful representations from the input data, while the decoder tries to reconstruct the original input.
Classification:

A classifier is trained on the encoded features to predict categorical targets. It operates on the learned representations from the encoder.

Test-Time Adaptation (TTA):

During testing, the autoencoder is fine-tuned on test data without labels. This adaptation helps the model adjust to the distribution shift between the training and test data.
The classifier uses these adapted representations to make more accurate predictions.
Test-Time Training:

The autoencoder continues learning even during the testing phase by trying to reconstruct the input test data. This ensures that the feature representations are better aligned with the test data's distribution.
Prediction on New Data:

The model predicts on new, unseen data after test-time adaptation, producing results that are stored for further analysis.

Overall Goal:
The approach aims to improve generalization performance by dynamically adapting the model at test time, helping it handle distributional shifts in real-world data more effectively.

