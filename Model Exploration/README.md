# Model Exploration

This section of code pertains to dataset creation and model exploration.

Data creation/create_ml_dataset.ipynb merges the final event dataset with the meteor frequency data to get a final dataset of 4.7 million rows.


MLPModelExploration/Model_Exploration.ipynb: This file contains the MLP (Multi-Layer Perceptron) Model code. This model takes in the 2 input channels and uses them to 
classify the event into one of 3 categories. The model is trained on the custom dataset we created and was trained multiple times in an attempt to increase the precision.

KNN and CNN Classifiers - 
    data_processing/B_FFT_Graph_Generator/FFT_Generator_image.py is similar to the FFT_Generator code that was originally created by UW SEAL but changes the PNGs saved to remove the axes and titles so they can be used as image arrays for the CNN model.

    KNN_CNN_Classifier.ipynb - Jupyter notebook file that has KNN and CNN classifier code. 


