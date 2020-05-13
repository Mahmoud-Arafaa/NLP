import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,classification_report
import re
import nltk




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




def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    X_indices = np.zeros((m,max_len))
    
    for i in range(m):                               
        
        sentence_words = [i.lower() for i in X[i].split()]
        
        j = 0
        
        for w in sentence_words:
            if(checkKey(word_to_index,w)==1):
                
                X_indices[i, j] = word_to_index[w]
            else :
                X_indices[i,j]= 0
          
            j = j+1
    return X_indices




def wordToIndex(dataFile2):
    dic={}
    counter=1
    for key in dataFile2:
      dic[key] = counter
      counter+=1
    return dic




def checkKey(dict, key): 
      
    if key in dict.keys(): 
       return 1
    else: 
        return 0



def IndexToWord(dataFile2):
    dic={}
    counter=1
    for key in dataFile2:
      dic[counter] = key
      counter+=1
    return dic





def deleteElement(dataToList):
    for l in dataToList:
        del l[0]
    
    return dataToList
    




def clean_sent(sent):
    return " ".join(w for w in nltk.wordpunct_tokenize(sent) \
     if w.lower() in words or not w.isalpha())         
        



def get_emp_matrix(word_to_index, word_to_vec_map):
    
    vocab_len = len(word_to_index) + 1                  
    emb_dim = word_to_vec_map["cucumber"].shape[0]      
    emb_matrix = np.zeros((vocab_len, emb_dim))
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    return emb_matrix





def InputSentence(inp):
    empty=[]
    empty.append(inp)
    empty = np.asarray(empty)
    new_indices=sentences_to_indices(empty, wTOi, max_len=40)
    new_indices = pad_sequences(new_indices, maxlen=40, dtype='int32', value=0)
    sentiment = model.predict(new_indices,batch_size=1,verbose = 3)[0]
    if(np.argmax(sentiment) == 0):
        print("negative")
    elif (np.argmax(sentiment) == 1):
        print("positive")
    elif(np.argmax(sentiment) == 2):
        print("neutral")
    

    
path = "data.txt"
dataFile2 = load_embeddings(path)
wTOi=wordToIndex(dataFile2)
iTow=IndexToWord(dataFile2)



words = set(nltk.corpus.words.words())
dataFrame = pd.read_csv(r'Tweets.csv',encoding='utf-8')
dataFrame = dataFrame[['airline_sentiment','text']]
dataFrame['airline_sentiment']= dataFrame['airline_sentiment'].map({'neutral':2,'negative':0,'positive':1})
dataFrame['text'] = dataFrame['text'].map(lambda x:re.sub('@\w*','',str(x)))          
dataFrame['text'] = dataFrame['text'].map(lambda x:re.sub('[^a-zA-Z]',' ',str(x)))     
dataFrame['text'] = dataFrame['text'].map(lambda x:re.sub('http.*','',str(x)))         
dataFrame['text'] = dataFrame['text'].map(lambda x:str(x).lower())
dataFrame['text'] = dataFrame['text'].apply(clean_sent)
Modefiyed=dataFrame['text']
dataToList=Modefiyed.reset_index().values.tolist()
dataToList=deleteElement(dataToList)
dataToList=np.array(dataToList)
dataToList=dataToList.reshape((14640,))
X1_indices = sentences_to_indices(dataToList,wTOi, max_len = 40)
emb_matrix = get_emp_matrix(wTOi, dataFile2)




numOFWoeds = len(dataFile2)+1
embed_dim = 50
lstm_out = 196
model = Sequential()
model.add(Embedding(numOFWoeds, embed_dim,input_length = 40,weights=[emb_matrix], trainable=False))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())




Y = pd.get_dummies(dataFrame['airline_sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X1_indices,Y, test_size = 0.20, random_state = 42)
batch_size = 128
model.fit(X_train, Y_train, epochs = 20 , batch_size=batch_size, verbose = 1)
Y_pred = model.predict_classes(X_test,batch_size = batch_size)
df_test = pd.DataFrame({'true': Y_test.tolist(), 'pred':Y_pred})
df_test['true'] = df_test['true'].apply(lambda x: np.argmax(x))

  
      
inp=input("Enter Input: ")
InputSentence(inp)