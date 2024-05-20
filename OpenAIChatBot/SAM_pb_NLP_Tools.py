import openai
import os 
import time 
from crewai import Agent, Task, Crew, Process 
from crewai_tools import SerperDevTool, WebsiteSearchTool
from langchain_openai import ChatOpenAI
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from crewai_tools import tool
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeMethods:
    def urldatareader(self):
        '''
        This tool reads data from a URL and returns a pandas dataframe.
        '''
        url = input("Please submit your data URL: ")
        data = pd.read_csv(url)
        return data

    def CSVdatareader(self):
        '''
        This tool reads data from a CSV file and returns a pandas dataframe.
        '''
        file_path = input("Please submit your CSV file path: ")
        data = pd.read_csv(file_path)
        return data

    def CSV_or_URL(self):
        '''
        This tool asks the user if they want to use a CSV file or a URL for data.
        '''
        CSV_or_URL = ''
        
        while CSV_or_URL != 'CSV' and CSV_or_URL != 'URL':
            CSV_or_URL = input("Would you like to use a CSV file or a URL for your data? (CSV or URL): ")
            if CSV_or_URL == 'CSV':
                data = self.CSVdatareader()
            elif CSV_or_URL == 'URL':
                data = self.urldatareader()

        return data

    def variablecreator(self):
        '''
        This tool takes a user input on what the Y, or dependent variable, is.
        It then returns an X, Y, and data back.
        THIS METHOD CANNOT BE USED IF THERE ARE MULTIPLE DEPENDENT OR TARGET VARIABLES.
        '''
        data = self.CSV_or_URL()
        print("Here are the columns in your dataset: ")
        print(data.columns)
        time.sleep(2)
        Selected_Y = input("Now, please select your Y (dependent) variable from the above columns: ")
        X = data.drop(Selected_Y, axis=1)
        Y = data[Selected_Y]
        return data, X, Y

    def trainingandtestingdata(self):
        '''
        This tool takes in the data, X, and Y and splits it into training and testing data.
        Returning the X_train, X_test, Y_train, and Y_test variables for later algorithm trading. 
        '''
        data, X, Y = self.variablecreator()
        test_size_question = float(input("Pick a number between 0 and 1 which represents what percent of the data will be used to test on: "))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size_question, random_state=37)
        return X_train, X_test, Y_train, Y_test

    def decisiontreeclassifier(self):
        '''
        This tool is called and returns the classifier for use in the prediction tool.
        '''
        X_train, X_test, Y_train, Y_test = self.trainingandtestingdata()
        model = DecisionTreeClassifier(max_depth=3) #max depth is the maximum number of levels in the tree, and can be optimized.
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        evaluate = input("Would you like to evaluate your model? (yes or no): ")
        
        while evaluate != 'yes' and evaluate != 'no':
            evaluate = input("Please enter 'yes' or 'no': ")

        if evaluate == 'yes':
            accuracy = accuracy_score(Y_test, Y_pred)
            confusion = confusion_matrix(Y_test, Y_pred)
            classification_rep = classification_report(Y_test, Y_pred)

            # Print the results
            print("\n--- Model Performance Metrics ---")
            print(f"Accuracy: {accuracy:.2f}")
            print("Confusion Matrix:")
            print(confusion)
            print("Classification Report:")
            print(classification_rep)

            # Visualize the decision tree
            plt.figure(figsize=(25, 10))
            plot_tree(model, 
                      filled=True, 
                      feature_names=['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price',
                                     'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order'],
                      class_names=['Non-Fraud', 'Fraud'])
            plt.show()
        elif evaluate == 'no':
            return Y_pred

def run_method_by_name(obj, method_name):
    try:
        method = getattr(obj, method_name)
        if callable(method):
            method()
        else:
            print(f"'{method_name}' is not callable")
    except AttributeError:
        print(f"'{method_name}' method not found")