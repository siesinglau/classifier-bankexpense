### functions and imports ###
import tensorflow as tf
import numpy as np
from tensorflow import keras
from termcolor import colored

EPOCHS=300

# main variable to control how this program will behave
def get_run_type():
    print('.....................................................................................................')
    print('Enter 1 to build the learning model from scratch, train it and then save it.')
    print('Enter 2 to load an earlier trained model and just retrain it with additional data.')
    print('Enter 3 to load an earlier trained model and use it to predict the categories from an input csv file.')
    print('Enter 8 to display all category labels')
    print('Enter 9 to exit')
    print('.....................................................................................................')

    run_type = input('Please select how you want to run this program.')

    return run_type
                 
# category label lookup codes
lookup = {'investment':1,'rent':2,'education':3,'groceries':4,'travel':5,'home':6,'restaurants':7,'kids':8,
         'insurance':9,'petrol':10,'medicine':11,'shopping':12,'utilities':13,'gifting':14,
          'car':15,'others':16,'entertainment':17,'transportation':0}
reverse_lookup = {i:w for w, i in lookup.items()}

#function to remove symbols and numbers from the input string
def cleantext(string):
    for char in '0123456789!@#$%^&*()-=+_,./<>?;:':
        string = string.replace(char,'')
    #remove double spaces
    string = string.replace('  ',' ')
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
    f = open(filename, 'r')
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
    
#class that contains the wordindex dictionary
class WordIndex:
    
    def __init__(self, filename):
        
        #read word index file when instantiate
        self.filename = filename
        wi = open(self.filename,'r+')
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
        
        #first parse input data into a unique word list 
        inputwords = list()
        for line in self.inputdata:
            for word in line.split():
                inputwords.append(word)
        uniqwords = set(inputwords)
        
        #then check what is the highest index currently in the word index.
        #the next new word will take the next number and added to the wordindex file
        wi = open(self.filename,'a')
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
    print('Using Machine Learning to learn and predict expense categories\n')
    print('Some Instructions:')
    print('When training the model, the input file should be in CSV format and named as training.csv')
    print('The 3rd column of of the input file contains a text description of the expense preferably in less than 8 words.')
    print('The 4th column of the input file should contain the text of one of the 18 expense categories corresponding to')
    print('the text description.\n')
    print('When predicting using the model, the input file should be in CSV format and named as predict.csv')
    print('The 3rd column of of the input file contains a text description of the expense preferably in less than 8 words.')
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

            #update wordindex dictionary and file if any new words are found
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

            #standardise length of training data to 8
            traindata = keras.preprocessing.sequence.pad_sequences(traindata, maxlen=8, padding='post', truncating='post', value = 0)

            if run_type == 1:
                #build the model
                model = keras.Sequential()
                model.add(keras.layers.Embedding(100000,32,input_length=8))
                model.add(keras.layers.GlobalAveragePooling1D())
                model.add(keras.layers.Dense(32, activation=tf.nn.relu))
                model.add(keras.layers.Dense(18, activation=tf.nn.softmax))

                #compile the model
                model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

                #train the model
                early_stop = keras.callbacks.EarlyStopping(monitor='acc', patience=30)
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

                for i,output in enumerate(predict):
                    if output[np.argmax(output)] * 100 < 90:
                        print(colored('{0:40} ==> {1:15} {2:10.2f}% probability'.format(inputdata[i], 
                                                                            label2text(int(np.argmax(output))), 
                                                                            output[np.argmax(output)]*100),'red'))
                    else:
                        print(colored('{0:40} ==> {1:15} {2:10.2f}% probability'.format(inputdata[i], 
                                                                            label2text(int(np.argmax(output))), 
                                                                            output[np.argmax(output)]*100),'green'))
            if run_type == 8:
                for i,w in lookup.items():
                    print(f"{w} ==> {i}")

        else:
            #exit when an input other than 1,2 or 3 is entered
            break
                        
    print('\nThanks for using the expense categorisation program')