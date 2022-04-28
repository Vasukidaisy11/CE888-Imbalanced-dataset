# CE888-Imbalanced-dataset

This study demonstrates how to deal with imbalanced datasets in a efficient way.  The prime objective of using datasets is to develop an accurate model  however, The model will be unreliable if the dataset is imbalanced. As a result, using imbalanced datasets will leads to unreliable and inaccurate model. The standard way of dealing with imbalanced dataset is to resampling the dataset. Either up sampling or down sampling the classes. But this project proposed a different way of dealing with imbalanced datasets with depends on supervised and unsupervised learning. In the actual world, datasets are used to make a larger number of medical judgments and analyses, the datasets model should produce more accurate models. Imbalanced datasets could potentially have a greater impact on real-world considerations. In this project , generating two different models using different validation method.  When both models are compared, it is clear when to perform which validation strategy. the stratified k fold technique used to validate the imbalance datasets. The stratified k fold cross-validation technique is a derivative of the cross-validation technique for classification issues. Throughout the K folds, it retains the same class ratio as the given dataset. And training the sample with random forest. Random forest is an ensemble learning-based supervised machine learning technique. The random forest algorithm mixes many decision trees of the same type. And also creating the model with K means and number of clusters identified from the elbow method. And calculating the findings using the right metrics
