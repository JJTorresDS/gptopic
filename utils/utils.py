import ollama
import google.generativeai as genai
from google_play_scraper import reviews, Sort
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras import regularizers

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import numpy as np
import pandas as pd
import nltk
import string

def get_google_play_reviews(app, count=100, filter_score_with=None,
                            lang="en", country="us", sort=Sort.NEWEST):
    """
    Wrapper of google_play_scrapper
    app: the code of the app you want to scan (eg: 'com.binance.dev')
    count: the number of reviews (defaults to 100)
    filter_score_with: if you want to filter reviews by number of stars
    """
    result, continuation_token = reviews(
            app,
            count=count, # defaults to 100
            filter_score_with=filter_score_with, # defaults to None(means all score)
            lang=lang, # defaults to 'en'
            country=country, # defaults to 'us'
            sort=sort # defaults to Sort.NEWEST
        )
    return result


def gemini_query(prompt, gemini_key,debug=False, counter=0, tries=3):
    
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    if counter < tries:
        try:
            response = model.generate_content(prompt)
        except: 
            
            if debug:
                print("Gemini Failed to respond. Sleeping...")
            time.sleep(10) 

            if debug:
                print("Entering recursive step.", counter+1)
            return gemini_query(prompt=prompt, 
                                gemini_key=gemini_key,
                                  counter=counter+1)
    else:
        return "gemini failed to respond"
    return response.text.strip()

def ollama_query(prompt, model = "llama3.2:1b"):
    #model = "deepseek-r1:7b"
    
    response = ollama.chat(
        model = model,
        messages = [{"role":"user", "content":prompt }]    
    )

    return response["message"]["content"].strip()

class TopicNeuralNet:
    """
        X: numpy array of text (ideally customer reviews)
        y: numpy array of text labels
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.avg_length = int(np.mean([len(r.split())for r in X]))
        self.median_length = int(np.median([len(r.split())for r in X]))
        self.max_length = int(np.max([len(r.split())for r in X]))
    ## bootstrapping
    def bootstrap(self, factor=100, replacement=True): # ==> OK
        self.df = pd.DataFrame({"X":self.X, "y":self.y})
        if replacement:
            self.df = self.df.sample(self.df.shape[0] * factor, replace=replacement)
            
        self.df = self.df.reset_index(drop=True)
        print(f"Bootstrapping by a factor of {factor}")
        return True
        
    ## preprocessing

    def text_normalizer(self,doc, preprocess):
        doc = doc.lower()
        doc = doc.strip()
        tokens = nltk.word_tokenize(doc)
        if preprocess:
            self.preprocess=True
            return [t for t in tokens if \
                t not in string.punctuation and \
                t not in self.stopwords ] #only consider alphanumeric character
        else:
            self.preprocess=False
            return tokens

    
    def text_preprocessing(self, text_array, preprocess):
        self.stopwords = nltk.corpus.stopwords.words("english")
        self.stemmer = nltk.stem.PorterStemmer()
        #text_preprocessed = self.df["X"].apply(lambda x: " ".join(self.text_normalizer(x)))
        text_array_processed = []
        corpus_preprocessed = []
        for text in text_array:
            processed_text = " ".join(self.text_normalizer(text, preprocess))
            text_array_processed.append(processed_text)
            corpus_preprocessed.extend(processed_text.split())

        self.counter_preprocessed = Counter(corpus_preprocessed)
        print("Normalizing text")
        return text_array_processed

    def text_tokenizer_fit(self):
        self.num_unique_words = len(self.counter_preprocessed)
        self.tokenizer = Tokenizer(num_words=self.num_unique_words)
        self.tokenizer.fit_on_texts(self.df.X_preprocessed)
        print("Fitting text tokenizer")

    def label_tokenizer_fit(self):
        self.le = LabelEncoder()
        self.le.fit(self.df.y)
        print("Fitting label tokenizer") 

    def text_tokenizer_transform(self, text_array):
        print("Tokenizing text")
        return self.tokenizer.texts_to_sequences(text_array)

    def label_tokenizer_transform(self, label_array):
        print("Tokenizing labels")
        return self.le.transform(label_array)

    def sequence_padding(self, text_sequences):
        text_sequences_padded = pad_sequences(text_sequences, maxlen= self.avg_length, padding = "post", truncating = "post")
        
        print("Passing text sequences")

        return text_sequences_padded

    def set_nn_architecture(self, num_classes, lstm_layers=3, multiplier=5):
        
        self.model = keras.Sequential()
        self.model.add(layers.Embedding(self.num_unique_words, num_classes*100))
        
        for i in range(lstm_layers):
            #self.model.add(layers.Dense(num_classes*20, activation='relu'))
            self.model.add(layers.LSTM(num_classes * multiplier, return_sequences=True))
            self.model.add(layers.Dropout(.5))
            
        
        self.model.add(layers.LSTM(num_classes * multiplier, return_sequences=False))
        self.model.add(layers.Dropout(.3))
        
        self.model.add(layers.Dense(num_classes*3, activation='relu'))
        self.model.add(layers.Dense(num_classes, activation="softmax"))
        
        print("Setting model architecture")

    def compile_nn(self):
        #loss and optimizer
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        optim = keras.optimizers.Adam(learning_rate=0.001)
        metrics  = ["accuracy"]
        
        self.model.compile(optimizer=optim, loss=loss, metrics=metrics)
        print("Compiled model")
    
    def fit(self, bootstrapping=True, factor=100, preprocess=True, epochs=20):
        #bootstraping
        if bootstrapping:
            self.bootstrap(factor=factor)
        else:
            self.bootstrap(factor=1, replacement=False)

        #proprocess text
        self.df["X_preprocessed"] = self.text_preprocessing(self.df.X, preprocess=preprocess)

        #fitting tokenizers
        self.text_tokenizer_fit()
        self.label_tokenizer_fit()

        #tokenizing
        self.text_sequences = self.text_tokenizer_transform(self.df.X_preprocessed)
        self.label_sequences = self.label_tokenizer_transform(self.df.y)

        #padding
        self.text_sequences_padded = self.sequence_padding(self.text_sequences)

        #fit model
        num_classes = self.df.y.nunique()

        #set architecture
        self.set_nn_architecture(num_classes)

        #compile nn
        self.compile_nn()

        #implement early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            verbose=2
        )

        #training NeuralNet
        history = self.model.fit(self.text_sequences_padded,
            self.label_sequences,
            callbacks = [early_stopping],
            epochs=epochs,
            batch_size=512,
            validation_split=0.85)

    def predict(self, X_new=[1]):
        
        if len(X_new)>1:
            #proprocess text
            new_x = self.text_preprocessing(X_new, preprocess=self.preprocess)  
            #tokenize reviews
            new_x = self.text_tokenizer_transform(new_x)
            #padding
            new_x = self.sequence_padding(new_x)
        else:
            new_x = self.text_sequences_padded

        y_hats= self.model.predict(new_x)
        pred_list = []
        for p in y_hats:
           pred_list.append(p.argmax())
        return self.le.inverse_transform(pred_list)