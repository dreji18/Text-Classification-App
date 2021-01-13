# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 08:28:35 2021

@author: rejid4996
"""

# packages
import os
import re
import time
import base64
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
import preprocessor as p
from textblob.classifiers import NaiveBayesClassifier

# custum function to clean the dataset (combining tweet_preprocessor and reguar expression)
def clean_tweets(df):
    #set up punctuations we want to be replaced
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
    REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")
    tempArr = []
    for line in df:
        # send to tweet_processor
        tmpL = p.clean(line)
        # remove puctuation
        tmpL = REPLACE_NO_SPACE.sub("", tmpL.lower()) # convert all tweets to lower cases
        tmpL = REPLACE_WITH_SPACE.sub(" ", tmpL)
        tempArr.append(tmpL)
    return tempArr

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
            """Generates a link allowing the data in a given panda dataframe to be downloaded
            in:  dataframe
            out: href string
            """
            val = to_excel(df)
            b64 = base64.b64encode(val)  # val looks like b'...'
            return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="classified_data.xlsx">Download file</a>' # decode b'abc' => abc

def download_model(model):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="myClassifier.pkl">Download Model .pkl File</a>'
    st.markdown(href, unsafe_allow_html=True)

def main():
    """NLP App with Streamlit"""
    
    from PIL import Image
    
    wallpaper = Image.open('D 4 Data.jpg')
    wallpaper = wallpaper.resize((700,350))
        
    st.sidebar.title("Text Classification App 1.0")
    st.sidebar.success("Please reach out to https://www.linkedin.com/in/deepak-john-reji/ for more queries")
    st.sidebar.subheader("Classifier using Textblob ")
    
    st.info("For more contents subscribe to my Youtube Channel https://www.youtube.com/channel/UCgOwsx5injeaB_TKGsVD5GQ")
    st.image(wallpaper)
    
    options = ("Train the model", "Test the model", "Predict for a new data")
    a = st.sidebar.empty()
    value = a.radio("what do you wanna do", options, 0)
    
    if value == "Train the model":
        
        uploaded_file = st.file_uploader("*Upload your file, make sure you have a column for text that has to be classified and the label", type="xlsx")
        
        if uploaded_file:
    
            df = pd.read_excel(uploaded_file)
            
            option1 = st.sidebar.selectbox(
                    'Select the text column',
                    tuple(df.columns.to_list()))
            
            option2 = st.sidebar.selectbox(
                    'Select the label column',
                    tuple(df.columns.to_list()))
            
            # clean training data
            df[option1] = clean_tweets(df[option1])
            
            # Enter the label names 
            label1 = st.sidebar.text_input("Enter the label for '0' value") 
            label2 = st.sidebar.text_input("Enter the label for '1' value") 
            
            # replace value with pos and neg
            df[option2]  = df[option2].map({0:label1, 1:label2})
            
            gcr_config = st.sidebar.slider(label="choose the training size, longer the size longer the training time",
                               min_value=100,
                               max_value=10000,
                               step=10)
            
            #subsetting based on classes
            df1 = df[df[option2] == label1][0:int(gcr_config/2)]
            df2 = df[df[option2] == label2][0:int(gcr_config/2)]
            
            df_new = pd.concat([df1, df2]).reset_index(drop=True)
            
            
            # convert in the format
            training_list = []
            for i in df_new.index:
                value = (df_new[option1][i], df_new[option2][i])
                training_list.append(value)
            
            # run classification
            run_button = st.sidebar.button(label='Start Training')
            
            if run_button:
            
                # Train using Naive Bayes
                start = time.time()  # start time
                cl = NaiveBayesClassifier(training_list[0:gcr_config])
                
                st.success("Congratulations!!! Model trained successfully with an accuracy of "+str(cl.accuracy(training_list) * 100) + str("%"))
                st.write("Total Time taken for Training :" + str((time.time()-start)/60) + " minutes")
                              
                # download the model
                download_model(cl)
    
    # testing the model
    if value == "Test the model":  
        uploaded_file = st.file_uploader("*Upload your model file, make sure its in the right format (currently pickle file)", type="pkl")
        if uploaded_file:     
            model = pickle.load(uploaded_file)
            st.success("Congratulations!!! Model upload successfull")
            
            if model:
                value1 = ""
                test_sentence = st.text_input("Enter the testing sentence") 
                
                #predict_button = st.button(label='Predict')
                
                if test_sentence:
                    st.info("Model Prediction is : " + model.classify(test_sentence))
                    
                    "\n"
                    st.write("### 🎲 Help me train the model better. How is the prediction?")
                    "\n"
                    correct = st.checkbox("Correct")
                    wrong = st.checkbox("Incorrect")
                    
                    if correct:
                        st.success("Great!!! I am happy for you")
                        st.write("If you would like please try out for more examples")
                        
                    if wrong:
                        st.write("### 🎲 Dont worry!!! Lets add this new data to the model and retrain. ")
                        label = st.text_input("Could you write the actual label, please note the label name should be the same while you trained") 
                        #retrain_button = st.button(label='Retrain')
                        if label:
                            new_data = [(test_sentence, label)]
                            model.update(new_data)
                            
                            st.write("### 🎲 Lets classify and see whether model had learned from this example ")
                            
                            st.write("Sentence : " + test_sentence)
                            st.info("New Model Prediction is : " + model.classify(test_sentence))
                            
                            sec_wrong3 = st.checkbox("It's Correct")
                            sec_wrong1 = st.checkbox("Still Incorrect")
                            sec_wrong2 = st.checkbox("I will go ahead and change the data in excel and retrain the model")
                            
                            
                            if sec_wrong1:
                                st.write("### 🎲 Lets try training with some sentences of this sort")
                                new_sentence = st.text_input("Enter the training sentence") 
                                new_label = st.text_input("Enter the training label") 
                                
                                st.write("Lets try one last time ")
                                retrain_button1 = st.button(label='Retrain again!')
                                                                
                                if retrain_button1:
                                    new_data1 = [(new_sentence, new_label)]
                                    model.update(new_data1)
                                    
                                    st.write("Sentence : " + new_sentence)
                                    st.info("New Model Prediction is : " + model.classify(new_sentence))
                                    
                                    # download the model
                                    download_model(model)
                                    
                            if sec_wrong2:
                                st.info("Great!!! Fingers Crossed")
                                st.write("### 🎲 Please return to your excel file and add more sentences and Train the model again")
                                                            
                            if sec_wrong3:
                                st.info("Wow!!! Awesome")
                                st.write("Now lets download the updated model")
                                # download the model
                                download_model(model)
    
    # predicting for new data 
    if value == "Predict for a new data":  
        uploaded_file3 = st.file_uploader("*Upload your model file, make sure its in the right format (currently pickle file)", type="pkl")
        if uploaded_file3:     
            model1 = pickle.load(uploaded_file3)
            st.success("Congratulations!!! Model uploaded successfully")
            
            uploaded_file1 = st.file_uploader("*Upload your new data which you have to predict", type="xlsx")
            if uploaded_file1:     
                st.success("Congratulations!!! Data uploaded successfully")
                
                df_valid = pd.read_excel(uploaded_file1)
                
                option3 = st.selectbox(
                    'Select the text column which needs to be predicted',
                    tuple(df_valid.columns.to_list()))
                
                predict_button1 = st.button(label='Predict for new data')
                
                if predict_button1:
                    start1 = time.time()  # start time
                    df_valid['predicted'] = df_valid[option3].apply(lambda tweet: model1.classify(tweet))
                    
                    st.write("### 🎲 Prediction Successfull !!!")
                    
                    st.write("Total No. of sentences: "+ str(len(df_valid)))
                    st.write("Total Time taken for Prediction :" + str((time.time()-start1)/60) + " minutes")
                    
                    st.markdown(get_table_download_link(df_valid), unsafe_allow_html=True)
        
if __name__ == "__main__":
    main()
