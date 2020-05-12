import pandas as pd
import re
import nltk
import csv
import numpy as np
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split



def load_embeddings(path):
  glove_file = path
  with open(glove_file,encoding="utf8") as f:
    words = set()
    word_to_vec_map = {}
    for line in f:
      line = line.split(" ")
      curr_word = line[0]
      words.add(curr_word)
      word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
  return word_to_vec_map


def Return_Feature_vector(dataFile,XText):
    result=[]
    r=[]
    error=[]
    for sentenceList in XText:
        #print(sentenceList)
        for word_in_sentence in range(0,len(sentenceList)):
            try:
                value = dataFile[sentenceList[word_in_sentence]]
                r.append(list(value))
            except:
                error.append(0)
        result.append(list(np.array(r)))
        r=[]
        
    return result

def get_average(ListM):
    res=[]
    for List in ListM:
        
        res.append(np.average(List,0))
    
    return res

def deletElemnt(Nlist):
    size=0
    
    for list_small in Nlist:
        #print(type(Nlist[list_small]))
        if type(list_small) == np.float64:
            
            Nlist.pop(size)
        size+=1
    return Nlist


def deletElemntFromDataframe(Nlist,dataModefied):
    size=0
    for list_small in Nlist:
        #print(type(Nlist[list_small]))
        if type(list_small) == np.float64:
            #print(size)
            Nlist.pop(size)
            dataModefied.pop(size)
        size+=1
    return dataModefied

def textInput(texInput):
    texInput = texInput.split(" ")
    ListWords = [texInput]
    feature_vector_Input = Return_Feature_vector(dataFile,ListWords)
    feature_vector_Input = get_average(feature_vector_Input)
    a=logisticRegr.predict(feature_vector_Input)
    conversion = {1:"Positive",0:"Negative",-1:"Natural"}
    return conversion[a[0]]


path = "data.txt"
dataFile = load_embeddings(path)
dataFrame = pd.read_csv(r'Tweets.csv',encoding='utf-8')
dataFrame = dataFrame[['airline_sentiment','text']]                                    #To get just the two colomuns we need!
dataFrame['text'] = dataFrame['text'].map(lambda x:re.sub('@\w*','',str(x)))           #Removee any words starts with @ symbols
dataFrame['text'] = dataFrame['text'].map(lambda x:re.sub('[^a-zA-Z]',' ',str(x)))     #Remove special characters except [a-zA-Z]
dataFrame['text'] = dataFrame['text'].map(lambda x:re.sub('http.*','',str(x)))         #Remove link starts with https
dataFrame['text'] = dataFrame['text'].map(lambda x:str(x).lower())                     #Truned To small char's 
corpus = []
dataFrame['text']=dataFrame['text'].map(lambda x:corpus.append(' '.join([word for word in str(x).strip().split() if not word in set(stopwords.words('english'))])))
airlineSentiment = pd.DataFrame(data=corpus,columns=['text'])                                   
dataFrame['airline_sentiment']= dataFrame['airline_sentiment'].map({'neutral':-1,'negative':0,'positive':1})
dataFrame["text"] = airlineSentiment["text"].str.split(" ")
dataModified = dataFrame[["text","airline_sentiment"]]



getFeatureVector = Return_Feature_vector(dataFile,list(dataModified["text"]))
getAvarageVector = get_average(getFeatureVector) 
dataModified = deletElemntFromDataframe(getAvarageVector,list(dataModified["airline_sentiment"]))
getAvarageVector = deletElemnt(getAvarageVector)



df = pd.DataFrame(getAvarageVector, columns=None)
df["airline_sentiment"] = dataModified




logisticRegr = LogisticRegression()
cols = df.columns
cols = cols[0:len(list(cols))-1]
x = df[cols]




y=df["airline_sentiment"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42) 
log = logisticRegr.fit(x_train,y_train)



clf = svm.SVC()
svmTest = clf.fit(x_train,y_train)


print("logisticRegr :",log.score(x_test,y_test))
print("svm :",clf.score(x_test,y_test),"\n")




Input = input("Enter Input: ")
print(textInput(Input))
                                                    