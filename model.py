import streamlit as st
import tensorflow.keras.utils as ku
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

# Function to generate text using the built model
def generate_text(seed_text, model, tokenizer, max_sequence_len, next_words=25):
    output_text = seed_text

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word
        output_text += " " + output_word

    return output_text

# Loading the dataset
data = open('poem.txt', encoding="utf8").read()

# EDA: Generating WordCloud to visualize the text
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="black").generate(data)

# Streamlit app
st.title('Text Generation using Bi-Directional LSTM')
st.image(wordcloud.to_array(), use_container_width=True)

# Fitting the Tokenizer on the Corpus
corpus = data.lower().split("\n")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

# Vocabulary count of the corpus
total_words = len(tokenizer.word_index)

# Generating Embeddings/Vectorization
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
label = ku.to_categorical(label, num_classes=total_words+1)

# Building a Bi-Directional LSTM Model
model = Sequential()
model.add(Embedding(total_words+1, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150, return_sequences=True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words+1/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(total_words+1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training
history = model.fit(predictors, label, epochs=200, batch_size=32, verbose=1)

# Text generation interface
st.subheader('Text Generation Interface')
seed_text = st.text_input('Enter seed text:', 'Life begins')
generated_text = generate_text(seed_text, model, tokenizer, max_sequence_len, next_words=25)
st.write(generated_text)


