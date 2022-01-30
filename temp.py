import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats



from keras.datasets import imdb 
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential 
from keras.layers.embeddings import Embedding 
from keras.layers import SimpleRNN, Dense, Activation 





# call load_data with allow_pickle implicitly set to true
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(path = "imdb.npz",
                                                       num_words= None,
                                                       skip_top = 0,
                                                       maxlen = None,
                                                       seed = 113,
                                                       start_char = 1,
                                                       oov_char = 2,
                                                       index_from = 3)


print("Type: ", type(X_train))  #1
print("Type: ", type(Y_train))  #1

print("X train shape: ",X_train.shape) #1
print("Y train shape: ",Y_train.shape) #1


# %% Eda

print("Y train values: ",np.unique(Y_train)) #2 Accuracy değerleri belirleme
print("Y test values: ",np.unique(Y_test)) #2



unique, counts = np.unique(Y_train, return_counts = True) # #Y train distribution:  {0: 12500, 1: 12500} %100 dengeli bir veriseti
print("Y train distribution: ", dict(zip(unique,counts)))  #3

unique, counts = np.unique(Y_test, return_counts = True)
print("Y testdistribution: ",dict(zip(unique,counts)))


#visualization
plt.figure()  #4
sns.countplot(Y_train)
plt.xlabel("Classes")
plt.ylabel("Freg")
plt.title("Y train")


plt.figure() #4
sns.countplot(Y_test)
plt.xlabel("Classes")
plt.ylabel("Freg")
plt.title("Y test")


#kelime sayısı ve bunların dağılımlarını kontrol etme
d = X_train[0] #5 
print(d)
print(len(d))


review_len_train = [] 
review_len_test = []
for i,ii in zip(X_train,X_test): 
    review_len_train.append(len(i))
    review_len_test.append(len(ii))





sns.distplot(review_len_train, hist_kws = {"alpha":0.3}) #6
sns.distplot(review_len_test, hist_kws = {"alpha":0.3})  #6

print("Train mean:", np.mean(review_len_train))
print("Train median:", np.median(review_len_train))
print("Train mode:", stats.mode(review_len_train))


#word numbers
word_index = imdb.get_word_index() #7
print(type(word_index))
print(len(word_index))


#verilen sayıya göre hangi kelime olduğunu
for keys, values in word_index.items(): #8
    if values == 11111:
        print(keys)



#reviewlari text haline döndürme işlemi
def whatItSay(index = 24):  #9
    
    reverse_index = dict([(value,key) for (key, value) in word_index.items()])
    decode_review = " ".join([reverse_index.get(i - 3,"!") for i in X_train[index]])
    print(decode_review)
    print(Y_train[index])
    return decode_review


decoded_review = whatItSay(31)





# %% Preprocess

#call load_data
num_words = 15000 #10
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words = num_words)

maxlen = 150 #şimdilik 150 ile sınırlandırdık
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)


print(X_train[4])


for i in X_train[0:10]:
    print(len(i))
          
decoded_review = whatItSay(2)


# %% RNN

rnn = Sequential()
rnn.add(Embedding(num_words,32,input_length = len(X_train[0]))) 
rnn.add(SimpleRNN(16, input_shape = (num_words, maxlen), return_sequences = False, activation = "relu" )) #Sequential yapıma simpler Rnn eklendi
rnn.add(Dense(1)) 
rnn.add(Activation("sigmoid")) #act.fon. sigmoid binary classnification

print(rnn.summary()) 
rnn.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"]) 


history = rnn.fit(X_train, Y_train, validation_data= (X_test, Y_test), epochs=5, batch_size= 128, verbose=1) #Eğit


score = rnn.evaluate(X_test, Y_test) #accuracy/doğruluk hesabı 
print("Accuracy: %",score[1]*100)


#data görselleştirme
plt.figure()
plt.plot(history.history["accuracy"], label = "Train")
plt.plot(history.history["val_accuracy"], label = "Test")
plt.title("Acc")
plt.ylabel("Acc")
plt.xlabel("Epochs")
plt.legend()
plt.show()


plt.figure()
plt.plot(history.history["loss"], label = "Train")
plt.plot(history.history["val_loss"], label = "Test")
plt.title("Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()









