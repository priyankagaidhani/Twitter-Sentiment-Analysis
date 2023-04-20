import tweepy
import pandas as pd
import re
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import seaborn as sns
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import streamlit as st
from PIL import Image  # To display images
# nltk.download('wordnet')
plt.style.use('fivethirtyeight')


@st.cache_data(persist=True)
def load_data():
    df = pd.read_csv("Dataset.csv")
    df.head()
    return df


df = load_data()
# Removing unwanted columns : Author_ID, User_handle, Tweet_link
df = df.drop(columns=['No', 'Author_ID',
                      'Date_of_tweet', 'User_handle', 'Tweet_link'])
# Lowercasing the Tweets
df['Tweet'] = df['Tweet'].str.lower()

# Removing http links, special characters(@, #, ', +, -, !, comma etc.)
df['Clean_tweet'] = df['Tweet'].str.replace(
    "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", regex=True)

customStopwords = ['i', 'me', 'my', 'we', 'our', 'you', 'has', 'ha', 'he', 'him', 'his', 'she', 'her', 'it', 'the', 'was', 'what', 'which', 'who', 'whom', 'am', 'is', 'in', 'are', 'was', 'were', 'be', 'a', 'an', 'and', 'of',
                   'at', 'by', 'for', 'to', 'on', 'off', 'here', 'there', 'when', 'where', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'so', 'than', 's', 't', 'can', 'now', 'd', 'l', 'm', 'o', 're', 've', 'y']

# Tokenization
tokenizedTweet = df['Clean_tweet'].apply(lambda x: x.split())

# Sentence Without Stopwords
sentenceWithoutStopword = tokenizedTweet.apply(
    lambda sentence: [word for word in sentence if not word in customStopwords])

# Lemmatization
lemmatizer = WordNetLemmatizer()
tokenizedTweet = tokenizedTweet.apply(
    lambda sentence: [lemmatizer.lemmatize(w) for w in sentence])

# Removing Stopwords
sentenceWithoutStopword = tokenizedTweet.apply(
    lambda sentence: [word for word in sentence if not word in customStopwords])

# Combining all words into single sentence
for i in range(len(sentenceWithoutStopword)):
    sentenceWithoutStopword[i] = " ".join(sentenceWithoutStopword[i])

df['Clean_tweet'] = sentenceWithoutStopword

# Create function to get subjectivity


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Create function to get polarity


def getPolarity(text):
    return TextBlob(text).sentiment.polarity


# Create 2 new colums
df['Subjectivity'] = df['Clean_tweet'].apply(getSubjectivity)
df['Polarity'] = df['Clean_tweet'].apply(getPolarity)

# Create function for -ve,neutral and +ve analysis


def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


df['Analysis'] = df['Polarity'].apply(getAnalysis)


def frequent_words():
    # Visualize frequent words
    allWords = " ".join([sentence for sentence in df['Clean_tweet']])

    wordCloud = WordCloud(width=500, height=300, random_state=30,
                          max_font_size=100).generate(allWords)

    plt.rcParams["figure.figsize"] = (3, 3)
    fig, x = plt.subplots()
    x.imshow(wordCloud, interpolation="bilinear")
    x.axis('off')
    st.pyplot(fig)


def polarity():
    #  replace with your desired colors
    plt.rcParams["figure.figsize"] = (5, 5)
    fig, ax = plt.subplots()
    colors = ["#3ED625", "#FFBF00", "#cc3232"]
    sns.set_palette(sns.color_palette(colors))
    sns.scatterplot(x="Polarity", y="Subjectivity",
                    hue="Analysis", data=df, ax=ax)
    plt.legend(loc='lower right')
    st.pyplot(fig)


def polarity_score():
    #  Visualising the Label count
    plt.rcParams["figure.figsize"] = (3, 3)
    fig, ax = plt.subplots()
    sns.countplot(x=df["Analysis"], ax=ax)
    st.pyplot(fig)


# Total people count : calculated from likes column
totalLikes = df.shape[0]
# df['Retweet-Count'].sum(axis = 0, skipna = True)

# percentage of +ve tweets
posTweets = df[df.Analysis == 'Positive']
# likesCountPos = posTweets['Retweet-Count'].sum(axis = 0, skipna = True)
likesCountPos = posTweets.count()
poCount = round((likesCountPos / totalLikes) * 100, 1)

# percentage of -ve tweets
negTweets = df[df.Analysis == 'Negative']
likesCountNeg = negTweets.count()
negCount = round((likesCountNeg / totalLikes) * 100, 1)

# percentage of neutral tweets
neutralCount = 100 - (poCount + negCount)

# print("positve = ", poCount, "% , Negative = ",
#       negCount, "% , Neutral = ", neutralCount)


def overall_sentiment():
    #  Plots pie chart
    value_counts = df['Analysis'].value_counts()
    # Create pie chart
    fig, ax = plt.subplots()
    ax.pie(value_counts, labels=value_counts.index, autopct='%1.0f%%')
    # Display pie chart
    st.pyplot(fig)


# Show value count
# df['Analysis'].value_counts()

# #  Visualising the Label count
# fig, ax = plt.subplots()
# sns.countplot(x=df["Analysis"], ax=ax)
# plt.title("Polarity Scores")
# st.pyplot(fig)


def frequent_positive_words():
    # Visualize frequent +ve words
    fig, ax = plt.subplots()
    allWords = " ".join(
        [sentence for sentence in df['Clean_tweet'][df['Analysis'] == 'Positive']])
    wordCloud = WordCloud(width=500, height=300, random_state=30,
                          max_font_size=100).generate(allWords)
    ax.imshow(wordCloud, interpolation="bilinear")
    ax.axis('off')
    st.pyplot(fig)


def frequent_negative_words():
    # Visualize frequent -ve words
    fig, ax = plt.subplots()
    allWords = " ".join(
        [sentence for sentence in df['Clean_tweet'][df['Analysis'] == 'Negative']])
    wordCloud = WordCloud(width=500, height=300, random_state=30,
                          max_font_size=100).generate(allWords)
    ax.imshow(wordCloud, interpolation="bilinear")
    ax.axis('off')
    st.pyplot(fig)


def frequent_neutral_words():
    # Visualize frequent neutral words
    fig, ax = plt.subplots()
    allWords = " ".join(
        [sentence for sentence in df['Clean_tweet'][df['Analysis'] == 'Neutral']])
    wordCloud = WordCloud(width=500, height=300, random_state=30,
                          max_font_size=100).generate(allWords)
    ax.imshow(wordCloud, interpolation="bilinear")
    ax.axis('off')
    st.pyplot(fig)

# Extracting hashtags


def hastag_extract(tweets):
    hashtags = []
    for tweet in tweets:
        ht = re.findall(r"#(\w+)", tweet)
        hashtags.append(ht)
    return hashtags


# Extract hashtags from +ve tweets
htPos = hastag_extract(df['Tweet'][df['Analysis'] == 'Positive'])
# Extract hashtags from -ve tweets
htNeg = hastag_extract(df['Tweet'][df['Analysis'] == 'Negative'])
# unnest list
htPos = sum(htPos, [])
htNeg = sum(htNeg, [])
freqPos = nltk.FreqDist(htPos)
dPos = pd.DataFrame({'Hashtag': list(freqPos.keys()),
                    'Count': list(freqPos.values())})


def positive_hashtags(dPos):
    # Selecting top 10 hashtags
    dPos = dPos.nlargest(columns='Count', n=5)
    plt.rcParams["figure.figsize"] = (15, 9)
    fig, ax = plt.subplots()
    sns.barplot(data=dPos, x='Hashtag', y='Count', ax=ax)
    st.pyplot(fig)


freqNeg = nltk.FreqDist(htNeg)
dNeg = pd.DataFrame({'Hashtag': list(freqNeg.keys()),
                    'Count': list(freqNeg.values())})


def negative_hashtag(dNeg):
    # Selecting top 10 hashtags
    dNeg = dNeg.nlargest(columns='Count', n=5)
    plt.rcParams["figure.figsize"] = (15, 9)
    fig, ax = plt.subplots()
    sns.barplot(data=dNeg, x='Hashtag', y='Count', ax=ax)
    st.pyplot(fig)


# Creating new dataframe which only has +ve and -ve values
dfAnalysis = df[df.Analysis != 'Neutral']

bowVectorizer = CountVectorizer(
    max_df=0.90, min_df=2, max_features=1000)
bow = bowVectorizer.fit_transform(dfAnalysis['Clean_tweet'])

# Creating a new label column : -ve as 1 , +ve as 0


def getLabel(score):
    if score < 0:
        return 1
    else:
        return 0


dfAnalysis.loc[:, 'Label'] = dfAnalysis['Polarity'].apply(getLabel)

xTrain, xTest, yTrain, yTest = train_test_split(
    bow, dfAnalysis['Label'], random_state=42, test_size=0.25)

# Training
model = LogisticRegression(max_iter=10000)
model.fit(xTrain, yTrain)

# Testing
yPred = model.predict(xTest)
f1_score(yTest, yPred)

accuracy_score(yTest, yPred)

# open a file, where you want to store the data

filename = 'sentimentfile'
outfile = open(filename, 'wb')

# dump information to that file
pickle.dump(model, outfile)
outfile.close()

modelReceieved = open(filename, 'rb')
newFile = pickle.load(modelReceieved)

# print(newFile)

# print(type(newFile))

y_prediction = newFile.predict(xTest)
f1_score(yTest, y_prediction)
# print("Logistic Regression accuracy: " +
#       str(accuracy_score(yTest, y_prediction)))

# Training
model = BernoulliNB()
model.fit(xTrain, yTrain)

# Testing
yPred = model.predict(xTest)
f1_score(yTest, yPred)

# print("Bernoulli Naive Bayes accuracy: " +
#       str(accuracy_score(yTest, yPred)))

# Training
model = LinearSVC(max_iter=10000)
model.fit(xTrain, yTrain)

# Testing
yPred = model.predict(xTest)
f1_score(yTest, yPred)

# print("SVM accuracy: "+str(accuracy_score(yTest, yPred)))

# Training
model = RandomForestClassifier(n_estimators=100)
model.fit(xTrain, yTrain)

# Testing
yPred = model.predict(xTest)
f1_score(yTest, yPred)

# print("Random Forest accuracy: "+str(accuracy_score(yTest, yPred)))

# sns.displot(yTest - yPred)
# plt.show()

df.to_csv('Final_Dataset.csv')


def main():  # beginning of the statement execution
    html_temp = """
    <div style="background-image: url(https://www.uwindsor.ca/drama/sites/uwindsor.ca.drama/files/twitter_logo.jpeg);
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
    padding:150px">
    <h2 style="color:blue;text-align:center;">Feelix<br>Twitter Sentiment Analysis</h2>
    </div>




    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("Team: Data Group 5")
    st.text("Mentor: Aakash Gupta")
    if st.sidebar.checkbox("The Four Golden Questions"):
        st.subheader(
            "1. What is the problem?")
        st.text(
            'The problem is that still now there are many product owners, politicians, actors,\netc who are failing because they do not have the candid insights of their target\naudience to make informed decisions and strategies.')
        st.subheader(
            "2. Whose problem are we solving?")
        st.text(
            'We are solving the problem of any organization, person, or product whose success is\ndetermined by customer satisfaction. Some examples in different categories would\nbe the product iPhone, as a personality may be a politician or an actor and so on.')
        st.subheader(
            "3. How do you the problem is solved?")
        st.text(
            'When we are providing through our project the candid sentiments of the target \naudience to the clients, and they are using that information to make informed\ndecisions and strategies and thereby succeed in the upcoming improved effort then \nwe know the problem is solved. And this is a continuous cycle.')
        st.subheader(
            "4. What is the business outcome of solving this problem?")
        st.text(
            'The business outcome is success for any and all organizations/individuals whose \nsvalue is determined by customer satisfaction.')
    if st.sidebar.checkbox("Twitter Sentiment Analysis"):
        st.subheader("Percentage wise sentiment representation")
        overall_sentiment()
    if st.sidebar.checkbox("Frequently used words"):
        st.subheader("Frequently used keywords")
        frequent_words()
        st.subheader("Frequently used positive keywords")
        frequent_positive_words()
        st.subheader("Frequently used negative keywords")
        frequent_negative_words()
        st.subheader("Frequently used neutral keywords")
        frequent_neutral_words()
    if st.sidebar.checkbox("Polarity Sentiment Analysis"):
        st.subheader("Subjectivity vs Polarity Scatter Plot")
        polarity()
        st.subheader("Polarity Scores")
        polarity_score()
    if st.sidebar.checkbox("Hashtag Sentiment Analysis"):
        st.subheader("Positive Hashtag Count")
        positive_hashtags(dPos)
        st.subheader("Negative Hashtag Count")
        negative_hashtag(dNeg)


if __name__ == '__main__':  # check for main executed when programme is called
    main()
