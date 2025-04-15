# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 05:36:18 2023

@author: thfar
"""

highschool = "C:/Users/thfar/OneDrive/Documents/Data Science/Random Stuff/Text Analytics Proj - Derek News/high school.txt" 
psu_sports = "C:/Users/thfar/OneDrive/Documents/Data Science/Random Stuff/Text Analytics Proj - Derek News/penn state-sports.txt"
psu_tele = "C:/Users/thfar/OneDrive/Documents/Data Science/Random Stuff/Text Analytics Proj - Derek News/penn state-tele.txt"
toledo = "C:/Users/thfar/OneDrive/Documents/Data Science/Random Stuff/Text Analytics Proj - Derek News/toledo.txt"

###############################################################################
#############3 Text Blob method ###############################################
############################################################################
from textblob import TextBlob


def analyze_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)

    # Get sentiment polarity (-1 to 1, where -1 is negative, 0 is neutral, and 1 is positive)
    sentiment_polarity = blob.sentiment.polarity

    # Classify sentiment as positive, negative, or neutral
    if sentiment_polarity > 0:
        sentiment = 'Positive'
    elif sentiment_polarity < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return sentiment, sentiment_polarity

def analyze_text_file(file_path):
    try:
        # Read the text from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Analyze sentiment
        sentiment, sentiment_polarity = analyze_sentiment(text)

        print(f"Text: {text[:100]}...")  # Display the first 100 characters of the text
        print(f"Sentiment: {sentiment}")
        print(f"Sentiment Polarity: {sentiment_polarity}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Replace 'your_text_file.txt' with the path to your text file
    analyze_text_file(highschool)
    
if __name__ == "__main__":
    # Replace 'your_text_file.txt' with the path to your text file
    analyze_text_file(psu_sports)
    
if __name__ == "__main__":
    # Replace 'your_text_file.txt' with the path to your text file
    analyze_text_file(psu_tele)
    
if __name__ == "__main__":
    # Replace 'your_text_file.txt' with the path to your text file
    analyze_text_file(toledo)


###################################################################################
############################33 VADER ###############################################
####################################################################################

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def analyze_text_file_nltk(file_path):
    try:
        # Read the text from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Create a SentimentIntensityAnalyzer object
        sid = SentimentIntensityAnalyzer()

        # Get sentiment scores
        scores = sid.polarity_scores(text)

        # Print the sentiment scores
        print(f"Text: {text[:100]}...")  # Display the first 100 characters of the text
        print(f"Sentiment: {'Positive' if scores['compound'] > 0 else 'Negative' if scores['compound'] < 0 else 'Neutral'}")
        print(f"Sentiment Polarity: {scores['compound']}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Replace 'your_text_file.txt' with the path to your text file
    analyze_text_file_nltk(highschool)
    
if __name__ == "__main__":
    # Replace 'your_text_file.txt' with the path to your text file
    analyze_text_file_nltk(psu_sports)
    
if __name__ == "__main__":
    # Replace 'your_text_file.txt' with the path to your text file
    analyze_text_file_nltk(psu_tele)
    
if __name__ == "__main__":
    # Replace 'your_text_file.txt' with the path to your text file
    analyze_text_file_nltk(toledo)


###############################################################################
############ Visualizing Sentiment Analysis ###############################
###############################################################################

import matplotlib.pyplot as plt
from textblob import TextBlob

def analyze_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)

    # Get sentiment polarity (-1 to 1, where -1 is negative, 0 is neutral, and 1 is positive)
    sentiment_polarity = blob.sentiment.polarity

    return sentiment_polarity

def visualize_sentiment_distribution(text):
    # Analyze sentiment for the entire text
    sentiment_polarity = analyze_sentiment(text)

    # Data for plotting
    labels = ['Negative', 'Neutral', 'Positive']
    sizes = [max(0, -sentiment_polarity), max(0, 1 - abs(sentiment_polarity)), max(0, sentiment_polarity)]  # Ensure non-negative values
    colors = ['red', 'gray', 'green']

    # Plotting
    plt.bar(labels, sizes, color=colors)
    plt.title(f'Sentiment Distribution\nOverall Sentiment Score: {sentiment_polarity:.2f}')
    plt.show()

if __name__ == "__main__":
    # Replace 'your_text_file.txt' with the path to your text file
    with open(highschool, 'r', encoding='utf-8') as file:
        text_content = file.read()

    visualize_sentiment_distribution(text_content)
    
if __name__ == "__main__":
    # Replace 'your_text_file.txt' with the path to your text file
    with open(psu_sports, 'r', encoding='utf-8') as file:
        text_content = file.read()

    visualize_sentiment_distribution(text_content)
    
if __name__ == "__main__":
    # Replace 'your_text_file.txt' with the path to your text file
    with open(psu_tele, 'r', encoding='utf-8') as file:
        text_content = file.read()

    visualize_sentiment_distribution(text_content)
    
if __name__ == "__main__":
    # Replace 'your_text_file.txt' with the path to your text file
    with open(toledo, 'r', encoding='utf-8') as file:
        text_content = file.read()

    visualize_sentiment_distribution(text_content)


################################################################################
########## General text analytics #############################################
################################################################################

import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

def analyze_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)

    # Get sentiment polarity (-1 to 1, where -1 is negative, 0 is neutral, and 1 is positive)
    sentiment_polarity = blob.sentiment.polarity

    return sentiment_polarity

def tokenize_and_remove_stopwords(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stopwords
    sww = stopwords.words('english')
    addtl_stopwords = ["'s", "'re", "n't", "24", "'m", "get"]
    for iii in addtl_stopwords:
        sww.append(iii)
    stop_words = set(sww)
    filtered_words = [word for word in words if word.lower() not in stop_words]

    return filtered_words

def analyze_word_frequency(text):
    # Tokenize and remove stopwords
    words = tokenize_and_remove_stopwords(text)

    # Calculate word frequency
    word_frequency = nltk.FreqDist(words)

    # Plotting
    word_frequency.plot(20, cumulative=False, title = "Word Frequency Plot: Toledo")
    plt.show()

if __name__ == "__main__":
    # Replace 'your_text_file.txt' with the path to your text file
    with open(toledo, 'r', encoding='utf-8') as file:
        text_content = file.read()

    # Sentiment analysis
    sentiment_polarity = analyze_sentiment(text_content)
    print(f"Sentiment Polarity: {sentiment_polarity:.2f}")

    # Word frequency analysis
    analyze_word_frequency(text_content)
    
if __name__ == "__main__":
    # Replace 'your_text_file.txt' with the path to your text file
    with open(psu_sports, 'r', encoding='utf-8') as file:
        text_content = file.read()

    # Sentiment analysis
    sentiment_polarity = analyze_sentiment(text_content)
    print(f"Sentiment Polarity: {sentiment_polarity:.2f}")

    # Word frequency analysis
    analyze_word_frequency(text_content)
    
if __name__ == "__main__":
    # Replace 'your_text_file.txt' with the path to your text file
    with open(psu_tele, 'r', encoding='utf-8') as file:
        text_content = file.read()

    # Sentiment analysis
    sentiment_polarity = analyze_sentiment(text_content)
    print(f"Sentiment Polarity: {sentiment_polarity:.2f}")

    # Word frequency analysis
    analyze_word_frequency(text_content)
    
if __name__ == "__main__":
    # Replace 'your_text_file.txt' with the path to your text file
    with open(toledo, 'r', encoding='utf-8') as file:
        text_content = file.read()

    # Sentiment analysis
    sentiment_polarity = analyze_sentiment(text_content)
    print(f"Sentiment Polarity: {sentiment_polarity:.2f}")

    # Word frequency analysis
    analyze_word_frequency(text_content)



###############################################################################
###################### Topic Clustering #######################################
###############################################################################

from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import string

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


def preprocess_text(text):
    # Tokenize and lemmatize the text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove stopwords and punctuation
    stop_words = set(STOPWORDS)
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]

    return tokens

def perform_topic_clustering(documents, num_topics=3, num_words=5):
    # Preprocess the documents
    processed_docs = [preprocess_text(doc) for doc in documents]

    # Create a dictionary and a corpus
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # Train the LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    # Print the topics
    for idx, topic in lda_model.print_topics(num_topics=num_topics, num_words=num_words):
        print(f'Topic {idx + 1}: {topic}')

    # Get the topic distribution for each document
    topic_distribution = [lda_model[doc] for doc in corpus]

    return topic_distribution

if __name__ == "__main__":
    # Replace 'file1.txt', 'file2.txt', 'file3.txt', 'file4.txt' with the paths to your text files
    file_paths = [highschool, psu_sports, psu_tele, toledo]
    documents = []

    # Read text from each file
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            documents.append(file.read())

    # Perform topic clustering for each document
    for i, doc in enumerate(documents):
        print(f"\nTopic Clustering for Document {i + 1}:\n")
        topic_distribution = perform_topic_clustering([doc])
        
        # Print the topic distribution for the current document
        for j, doc_topics in enumerate(topic_distribution):
            print(f"Document {i + 1} - Topic Distribution: {doc_topics}")



#######################################3
###############3 visualize ################
#############################################
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import nltk
import string
import matplotlib.pyplot as plt

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenize and lemmatize the text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove stopwords and punctuation
    stop_words = set(STOPWORDS)
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]

    return tokens

def perform_topic_clustering(documents, num_topics=3, num_words=5):
    # Preprocess the documents
    processed_docs = [preprocess_text(doc) for doc in documents]

    # Create a dictionary and a corpus
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # Train the LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    # Print the topics
    for idx, topic in lda_model.print_topics(num_topics=num_topics, num_words=num_words):
        print(f'Topic {idx + 1}: {topic}')

        # Create a word cloud for each topic
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(lda_model.show_topic(idx, topn=30)))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {idx + 1} Word Cloud')
        plt.show()

    # Get the topic distribution for each document
    topic_distribution = [lda_model[doc] for doc in corpus]

    return topic_distribution

if __name__ == "__main__":
    # Replace 'file1.txt', 'file2.txt', 'file3.txt', 'file4.txt' with the paths to your text files
    file_paths = [highschool, psu_sports, psu_tele, toledo]
    documents = []

    # Read text from each file
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            documents.append(file.read())

    # Perform topic clustering for each document
    for i, doc in enumerate(documents):
        print(f"\nTopic Clustering for Document {i + 1}:\n")
        topic_distribution = perform_topic_clustering([doc])
        
        # Print the topic distribution for the current document
        for j, doc_topics in enumerate(topic_distribution):
            print(f"Document {i + 1} - Topic Distribution: {doc_topics}")



