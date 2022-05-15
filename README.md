1. To run this program, type the following command
python3 NaiveBayes.py trainingfile testfile

2. This program implements Naive Bayes classifier and outputs true positive, false negative, false positive and true negative for training and test data.

3. key_val_split function stores labels and attributes of dataset.

4. data_get function reads train data and test data into labels and attributes.

5. train_model_px function computes probability of data given class

6. train_model function is the Naive Bayes training model to get the probabilities needed for prediction of labels. The model gives a  dictionary of labels and attributes of 4 levels. for label, for pro(y), pro(x), for attributes and for values.

7. llk_y function predicts labels with probability of label, and probability of data given label. 

---------------------------------------------------------------------------------------------------------------------------------