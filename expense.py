### This is a Python script that maps an expense text description to an expense category
### The mapping is achieved through a simple Tensorflow based neural network that is first trained
### with a set of training data supplied through a CSV file named training.csv.
### Once trained with sufficient data, the neural network model will be accurate enough to be used for prediction.
### The prediction is done by supplying a CSV file named predict.csv.
###
### Limitations:
### 1. The model is currently limited to be able to map to 20 expense categories
### 2. Only the first 8 words of the expense text description is used. The rest of the text is ignored.
### 3. The script creates a file called wordindex.txt which is a dictionary used to process the text description.
###    Currently the dictionary size is limited to 10,000 words.
###
### Customisation:
### 1. You can customise the expense categories by amending the data dictionary 'lookup' below


### functions and imports
import tensorflow as tf
import numpy as np
from tensorflow import keras
from termcolor import colored

EPOCHS=300
DICT_LIMIT = 10000

# main variable to control how this program will behave
def get_run_type():
    print('...................................O P T I O N S.....................................................')
    print('Enter 1 to build the learning model from scratch, train it and then save it.')
    print('Enter 2 to load an earlier trained model and just retrain it with additional data.')
    print('Enter 3 to load an earlier trained model and use it to predict the categories from an input csv file.')
    print('Enter 8 to display all category labels')
    print('Enter 9 to exit')
    print('.....................................................................................................')

    run_type = input('Please select how you want to run this program.')

    return run_type
                 
# category label lookup codes
lookup = {'investment':0,'rent or mortgage':1,'education':2,'groceries':3,'travel':4,'home maintenance':5,'restaurants':6,'kids expenses':7,
         'insurance':8,'petrol':9,'medicine':10,'shopping':11,'utilities':12,'gifting':13,
          'car maintenance':14,'entertainment':15,'transportation':16,'others':17}
reverse_lookup = {i:w for w, i in lookup.items()}

#function to remove symbols and numbers from the input string
def cleantext(string):
    for char in '0123456789!@#$%^&*()-=+_,./<>?;:':
        string = string.replace(char,'')
    #remove double spaces
    string = string.replace('  ',' ')
    #remove leading and trailing whitespace characters
    string = string.strip()
    return string.lower()

#function to lookup category text to labels
def text2label(text):
    
    return lookup[text.strip().lower()]

#function to lookup category labels back to text
def label2text(label):
    
    return reverse_lookup[label]

#function to read raw csv data file that contains text descriptions in column 3 and category in column 4.
#this is bank of america credit card Excel output format but category column is manually added.
def read_raw_data(filename):

    inputdata = list()
    outputdata = list()

    try:
        f = open(filename, 'r')
    except FileNotFoundError:
        print(f'ERROR: The file {filename} cannot be found. Please ensure that it is in the same folder as expense.py.')
        print('Check the sample file in https://github.com/siesinglau/classifier-bankexpense for the format needed.')
    else:
        for i,lines in enumerate(f):
            if i == 0:
                #always discard the header row
                pass
            else:
                lines = lines.split(',')
                desc = lines[2].replace('"','')
                inputdata.append(cleantext(desc))
                #skip output label if we are just running in predict mode
                if run_type != 3:
                    category = text2label(lines[3])
                    outputdata.append(category)
        f.close()
    
    return inputdata, outputdata

#function to save input index data to an input training file
def write_training_data(inputdata, filename):
    g = open(filename,'w')
    for i in inputdata:
        outputline = ''
        for j in i:
            outputline = outputline + str(j) + ', '
        g.write(outputline + '\n')
    g.close()
    
#function to write output labels into a file
def write_training_labels(outputdata, filename):
    g = open(filename,'w')
    for i in outputdata:
        g.write(str(i) + '\n')
    g.close()

def write_back_predict_results(predictoutput, predictsourcefile, outputfile):
    g = open(predictsourcefile, 'r')
    h = open(outputfile,'w+')
    for i, line in enumerate(g):
        #straight copy of the header row from the source predict.csv file to the output file
        if i == 0:
            h.write(line[:-1] + ',Predicted Expense Category,Prediction Probability\n')
        else:
            #write the predicted expense category with probability % at the end of the line
            h.write(line[:-1] + ',' + predictoutput[i-1]['category'] + ',probability=' + str(predictoutput[i-1]['probability']) + '\n')
    g.close()
    h.close()
    
#class that contains the wordindex dictionary
class WordIndex:
    
    def __init__(self, filename):
        
        self.filename = filename

        #read word index file when instantiate
        try:
            wi = open(self.filename,'r+')
        except FileNotFoundError:
            print('The word index file does not exist. A new index file will be created.')
            wi = open(self.filename,'x+')
        
        self.contents = dict()
        for line in wi:
            #strip out \n character
            line = line.replace('\n','')
            w,i = line.split(',')
            self.contents[w] = i
        wi.close()
        
        #contents is a dictionary mapping words to index, as per filename
        #contents_reverse is a dictionary mapping index to words
        self.contents_reverse = {i:w for w,i in self.contents.items()}
        

    def updateindex(self, inputdata, filename):
        
        #update word index if there are new words in inputdata. new words are then added to the wordindex file
        self.inputdata = inputdata
        self.filename = filename
        
        #only add to the wordindex file if the dictionary is still within its limit
        if len(self.contents.values()) < DICT_LIMIT:

            #first parse input data into a unique word list 
            inputwords = list()
            for line in self.inputdata:
                for word in line.split():
                    inputwords.append(word)
            uniqwords = set(inputwords)
            
            #then check what is the highest index currently in the word index.
            #the next new word will take the next number and added to the wordindex file
            wi = open(self.filename,'a')
            if len(self.contents.values()) == 0:
                highestindex = 0
            else:
                highestindex = max([int(i) for i in self.contents.values()])
            for word in uniqwords:
                try:
                    self.contents[word]
                except:
                    highestindex += 1
                    self.contents[word] = highestindex
                    wi.write(word + ', ' + str(highestindex) + '\n')
                else:
                    pass
            wi.close()
        
    def count_items(self):
        
        return len(self.contents)
    
    def get_index(self, word):
        
        return self.contents[word]
    
    def get_word(self, index):
        
        return self.contents_reverse[index]

class PrintProgress(keras.callbacks.Callback):
    def on_train_begin(self,logs):
        print('Model training starts...')

    def on_epoch_end(self, epoch, logs):
        progressind = ['-','/','|','\\','#']
        print(progressind[epoch%4],end='\b')
        if epoch % 5 == 0:
            print(progressind[4],end='')

    def on_train_end(self,logs):
        print('\nModel training completed.')

### this is the main program ###
if __name__ == "__main__":

    #some welcome words
    print('EXPENSE CATEGORISATION PROGRAM')
    print('------------------------------')
    print('Function:')
    print('Categorise expense text description into specific categories. See option 8 to view the current expense categories.')
    print('Program builds a simple neural network using Tensorflow Keras that can be trained to map the text description to the categories.')
    print('\n')
    print('Some Instructions:')
    print('To train the model, the input file should be in CSV format and named as training.csv')
    print('See training sample.csv for the formatting needed.')
    print('The 3rd column of of the input file contains a text description of the expense.')
    print('The 4th column of the input file contains the text of one of the expense categories.\n')
    print('When predicting using the model, the input file should be in CSV format and named as predict.csv')
    print('See predict sample.csv for the formatting needed.')
    print('The 3rd column of of the input file contains a text description of the expense.')
    print('The program will list as an output the predicted expense category for each expense text description.\n')

    while True:
        
        #first check how this program should be run
        run_type = get_run_type()

        while True:
            try:
                run_type = int(run_type)
            except ValueError:
                print('That was not a number. Please enter a selection of 1, 2, 3, 8 or 9\n')
                run_type = get_run_type()
            else:
                if run_type in [1,2,3,8,9]:
                    break
                else: 
                    print('That was not a valid selection. Please enter a selection of 1, 2, 3, 8 or 9\n')
                    run_type = get_run_type()

        if run_type in [1,2,3,8]:
            #read wordindex dictionary
            wordindex = WordIndex('wordindex.txt')

            #read raw data file and separate into input data and output labels
            if run_type == 3:
                inputdata, outputdata = read_raw_data('predict.csv')
            else:
                inputdata, outputdata = read_raw_data('training.csv')

            #update wordindex dictionary file if any new words are found
            wordindex.updateindex(inputdata,'wordindex.txt')

            #convert input data into indexes and save it to a training data file
            #due to the limitation of the model, the input text description is truncated at the 8th word
            indexdata = list()
            for line in inputdata:
                indexdata_row=list()
                for word in line.split():
                    indexdata_row.append(wordindex.get_index(word))
                indexdata.append(indexdata_row)

            if len(indexdata) > 0:
                write_training_data(indexdata,'traindata.txt')

            #if we are building or retraining the model, save output labels to a training labels file
            if len(outputdata) > 0:
                write_training_labels(outputdata,'trainlabels.txt')

            #convert lists to numpy arrays
            traindata = np.array(indexdata)
            trainlabels = np.array(outputdata)

            #standardise length of training data to 8 words
            traindata = keras.preprocessing.sequence.pad_sequences(traindata, maxlen=8, padding='post', truncating='post', value = 0)

            if run_type == 1:
                #build the model
                model = keras.Sequential()
                model.add(keras.layers.Embedding(DICT_LIMIT,32,input_length=8))
                model.add(keras.layers.GlobalAveragePooling1D())
                model.add(keras.layers.Dense(32, activation=tf.nn.relu))
                model.add(keras.layers.Dense(20, activation=tf.nn.softmax))

                #compile the model
                model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

                #train the model
                early_stop = keras.callbacks.EarlyStopping(monitor='acc', patience=10)
                history = model.fit(traindata, trainlabels, epochs=EPOCHS,
                        verbose=0, callbacks=[early_stop, PrintProgress()])

                #report out the final loss and accuracy values
                print(f"Final accuracy of model = {history.history['acc'][-1] * 100:{5}.{4}} %")
                print(f"Final loss value of model = {history.history['loss'][-1]:{5}.{4}}")

                #save the model with weights
                model.save('expensecategorisation.h5')

            if run_type == 2:
                #load previously saved model
                model = keras.models.load_model('expensecategorisation.h5')

                #retrain the model with new data
                early_stop = keras.callbacks.EarlyStopping(monitor='acc', patience=30)
                history = model.fit(traindata, trainlabels, epochs=EPOCHS,
                        verbose=0, callbacks=[early_stop, PrintProgress()])

                #report out the final loss and accuracy values
                print(f"Final accuracy of model = {history.history['acc'][-1] * 100:{5}.{4}} %")
                print(f"Final loss value of model = {history.history['loss'][-1]:{5}.{4}}")

                #save the model with weights
                model.save('expensecategorisation.h5')

            if run_type == 3:
                #load previously saved model
                model = keras.models.load_model('expensecategorisation.h5')

                #predict categories using input data
                predict = model.predict(traindata)

                predictoutput = list()
                for i,output in enumerate(predict):
                    
                    bestguess = dict()
                    bestguess['category'] = label2text(int(np.argmax(output)))
                    bestguess['probability'] = round(output[np.argmax(output)]*100,2)
                    predictoutput.append(bestguess)

                    if bestguess['probability'] < 90:
                        print(colored('{0:40} ==> {1:17} {2:10.2f}% probability'.format(inputdata[i][:40], 
                                                                            bestguess['category'], 
                                                                            bestguess['probability']),'red'))
                    else:
                        print(colored('{0:40} ==> {1:17} {2:10.2f}% probability'.format(inputdata[i][:40], 
                                                                            bestguess['category'], 
                                                                            bestguess['probability']),'green'))
                #write output to predict_results.csv
                write_back_predict_results(predictoutput,'predict.csv','predict_results.csv')

            if run_type == 8:
                print("These are the current expense categories:\n")
                for i,w in enumerate(reverse_lookup.values()):
                    print(f"{i+1}. {w}")

        else:
            #exit when an input other than 1,2 or 3 is entered
            break
                        
    print('\nThanks for using the expense categorisation program')