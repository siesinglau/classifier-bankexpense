# classifier-bankexpense
Tensorflow based classifier for bank expense description

Step 1: Install Python by following the instructions at https://realpython.com/installing-python/
Step 2: Install TensorFlow by following the instructions at https://www.tensorflow.org/install/
Step 3: Copy the files in this repository to a folder on your PC. Make sure they are all in the SAME folder.
Step 4: Launch a command prompt/terminal and go to the folder where the downloaded files are located.
Step 5: Run the python script by entering in the command prompt/terminal: python expense.py
Step 6: Enter option 8 to see the currently configured expense categories
Step 7: Enter option 1 to build a neural network model and train it using the dummy training data in training.csv
        The training.csv file contains dummy expense description matched to expense categories.
Step 8: Enter option 3 to use the model to predict the expense categories for the dummy descriptions in predict.csv
        Program will also copy predict.csv to a new file predict_results.csv where the predicted expense categories are added.
Step 9: You can correct the results in predict_results.csv if the prediction is wrong. Save the updated file as training.csv,
        overwriting the old training.csv file (you can back up the old file before overwritting)
Step 10: Enter option 2 to retrain the model with the corrected data in training.csv
Step 11: Enter option 9 to quit the program

You can amend the expense categories in the expense.py file as well as data contents in training.csv and predict.csv to suit your own expense data. Make sure that you have sufficient training data to get a good accuracy when training the model.
