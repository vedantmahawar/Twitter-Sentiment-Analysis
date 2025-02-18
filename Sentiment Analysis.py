# Imports pandas for dataframes 
import pandas 
# Imports wordcloud for data visualization
# You have to install wordcloud for this import statement and
# the following code to work
# Use this command
# py -m pip install wordcloud
# borrowed this library from kaggle
from wordcloud import WordCloud
# Imports matplotlib.pyplot to print out pre and post model visualizations
# Imports it as "plot" for ease of typing and sake of reading
import matplotlib.pyplot as plot
# Imports test train split to split the data for training and testing 
from sklearn.model_selection import train_test_split
# Imports logistic regression to create the logistic regression model
from sklearn.linear_model import LogisticRegression
# Imports tf-idf to convert the text so that the linear model can understand it
from sklearn.feature_extraction.text import TfidfVectorizer
# Imports a label encoder to encode the y/target values for the model
from sklearn.preprocessing import LabelEncoder
# Imports accuracy score to calculate the accuracy of the logistic regression model
from sklearn.metrics import accuracy_score
# Imports stop words to get stop words for the logistic regression model
# You have to install the library for this import statement to work
# Use this command
# py -m pip install stop-words
from stop_words import get_stop_words

# Reads in the dataset as a pandas dataframe
# When running the code on your computer you must 
# change the path to wherever it is stored on your computer
# Reads in two copies of the dataset as the pre-model visualizations will 
# change the way the dataset looks for the sake of visulizations
# The dataset is ~27000 datapoints of twitter tweets 
# It contains three columns (four but I will removed the fourth shortly)
# The columns are text (the tweet), selected_text (unique/signifigant values
# from the text), sentiment (the actual sentiment of the text, either positive,
# negative, or neutral)
# One of the reasons I picked this dataset was the high number of datapoints which 
# allows my model to be more accurate and prevent overfitting
tweets_df = pandas.read_csv("C:\\Users\\vmahawar\\OneDrive - Eastside" + 
        " Preparatory School\\2023-2024\\Winter\\Advanced Programming AI\\" + 
        "Final\\Tweets1.csv")
tweets_df_visualization = pandas.read_csv("C:\\Users\\vmahawar\\OneDrive - " + 
                                          "Eastside Preparatory School\\2023-2024\\" +
                                          "Winter\\Advanced Programming AI\\Final\\" + 
                                          "Tweets1.csv")


# Visualizes the data 
print(tweets_df)

# Drops the unneccssary "textID" column from the dataset
# This is unneccessary because it has no bearing on the sentiment of the text
# so therefore we do not need to include it into our model
# Drops the column "in place" leaving the dataframe as is, instead of 
# creating a copy
tweets_df.drop(columns = ["textID"], inplace = True)
tweets_df_visualization.drop(columns = ["textID"], inplace = True)

# Fills in all the N/A values for the text and selected_text column 
# these columns are going to be used for the actual linear model
# if the na values are not filled the model won't work
tweets_df["text"].fillna("", inplace = True)
tweets_df["selected_text"].fillna("", inplace = True)

# Created a grouped dataframe based off the sentiments of the text
# Sets the axis parameter to 0 which splits the data on rows (rather than columns)
# Sets group_keys to true which creates callable groups for each sentiment group
# I will use these to to create seperate dataframes for each group which then
# can be used for visualizations
# Sets dropna to true which drops NA values
# Once again, this is why I created seperate dataframes for the premodel-viz 
# and the actual regression model
# I had one singular dataframe for both before and it didn't work
grouped_df = tweets_df_visualization.groupby(by = "sentiment", axis = 0, 
                                             group_keys = True, dropna = True)

# Creates three dataframes for each group of each sentiment and prints
# one of them out to just visualize the structure 
# Creates these groups with the get_group command, which calls the groups
# we defined before 
# These dfs will be used for data visualization
positive_df = grouped_df.get_group("positive")
neutral_df = grouped_df.get_group("neutral")
negative_df = grouped_df.get_group("negative")
# Prints one of the grouped dataframes to just peek into what it looks
# like
print(positive_df)

# Drops the sentiment column from the dataframes so it doesn't throw off the 
# word cloud. I didn't do this before and the biggest word on the cloud was
# "positive" (or the other sentiments) and this didn't make sense to me 
# till I realized that the word cloud was utilizing the sentiment column which
# has positive every time
# Also does it in place 
positive_df.drop(columns = ["sentiment"], inplace = True)
neutral_df.drop(columns = ["sentiment"], inplace = True)
negative_df.drop(columns = ["sentiment"], inplace = True)

# Converts the dataframes above to a string so I can create visualizations 
# for them
# Does not index them as this data is uneccessary for the visualization 
positive_words = positive_df.to_string(index = False)
neutral_words = neutral_df.to_string(index = False)
negative_words = negative_df.to_string(index = False)

# Creates the word cloud for the positive sentiment
# Borrowed this visualization technique off kaggle
# This cloud is visualized with stop words so common English
# words will be the primary words (ex. and, I, is)
# This was to just visualize how the word cloud would look 
# without removing the stop words
positive_wordcloud_stop = WordCloud(width = 1500, height = 800, 
                                    background_color = 'white', min_font_size = 1, 
                                    stopwords = "").generate(positive_words)

# Displays the figure
# Set the size
plot.figure(figsize = (10, 10))
plot.imshow(positive_wordcloud_stop)
# Sets the title
plot.title("Word Cloud for Positive Tweets With Stopwords")
# Turns the axes off because they are unessecary, don't apply in this context,
# and without them the visualization is cleaner
plot.axis('off')
plot.show()

# Creates the word cloud for the positive sentiment
# without stopwords (if you remove the parameter it just uses
# the built in stopwords)
# Borrowed this visualization technique off kaggle
positive_wordcloud = WordCloud(width = 1500, height = 800, 
                                    background_color = 'white', min_font_size = 1
                                    ).generate(positive_words)
# Displays the figure
# Set the size
plot.figure(figsize = (10, 10))
plot.imshow(positive_wordcloud)
# Sets the title
plot.title("Word Cloud for Positive Tweets")
# Turns the axes off because they are unessecary, don't apply in this context,
# and without them the visualization is cleaner
plot.axis('off')
plot.show()

# Creates the word cloud for the neutral sentiment
# without stopwords (if you remove the parameter it just uses
# the built in stopwords)
# Borrowed this visualization technique off kaggle
neutral_wordcloud = WordCloud(width = 1500, height = 800, 
                                    background_color = 'white', min_font_size = 1
                                    ).generate(neutral_words)
# Displays the figure
# Set the size
plot.figure(figsize = (10, 10))
plot.imshow(neutral_wordcloud)
# Sets the title
plot.title("Word Cloud for Neutral Tweets")
# Turns the axes off because they are unessecary, don't apply in this context,
# and without them the visualization is cleaner
plot.axis('off')
plot.show()

# Creates the word cloud for the negative sentiment
# without stopwords (if you remove the parameter it just uses
# the built in stopwords)
# Borrowed this visualization technique off kaggle
negative_wordcloud = WordCloud(width = 1500, height = 800, 
                                    background_color = 'white', min_font_size = 1
                                    ).generate(negative_words)

# Displays the figure
# Set the size
plot.figure(figsize = (10, 10))
plot.imshow(negative_wordcloud)
# Sets the title
plot.title("Word Cloud for Negative Tweets")
# Turns the axes off because they are unessecary, don't apply in this context,
# and without them the visualization is cleaner
plot.axis('off')
plot.show()

# Visualizes the text data to just see what it looks like 
print(tweets_df)

# Splits the data for training and test 
# Utilizes a random_state to have reproducibility when it comes to accuracy
# This was just a random number I picked
X_train, X_test, Y_train, Y_test = train_test_split(
        tweets_df['selected_text'], tweets_df['sentiment'], random_state = 34)

# For the above command
# Originally the X_train data was set to the "text" column of the dataset
# This contains the entire tweet
# When I used this to perform sentiment analysis I had an accuracy of 68.43% 
# To tune my model I switched to the "selected_text" column of the dataset 
# This column contains a portion of the tweet that was selected by the dataset creator
# These usually contained more relavent and unique words and this improved my accuracy
# becasue these words were more likely to correlate with the sentiment 
# (because of their uniqueness)
# This boosted by accuracy to 82.86%, or by 14.43% which is extremely substantial 

# Old test/train split code that uses the "text" column of the dataset
# X_train, X_test, Y_train, Y_test = train_test_split(
#         tweets_df['text'], tweets_df['sentiment'], random_state = 34)

# Gets the stop words from the library 
# Has a parameter that says english becasue the library has 
# multi-language support 
stopwords = get_stop_words('english')

# Creates a instance of tfidf 
# Tf-idf stands for term frequency-inverse document frequencey
# It calculates how relavent a word is to a document based on two features
# Firstly, term frequency or simply how many times a term pops up in a document
# Secondly, inverse document frequency or how rare/common a word is in a set 
# of documents
# It them multiplies these features to calculate how relavent a term is
# For example words like "it" might pop up alot in a document, but when it 
# looks at the idf of the word "it" it will have a low idf (closer to 0 the 
# more common a word is)
# Because of this the high tf and low idf will cancel out and the word "it" 
# would seem less meaningful
# Compare this to a word like "car", say "car" has a high tf in a document and 
# since car is an uncommon word it would have a high idf
# These combine will place car in high importance within the document
# And this makes sense, it is a uncommon word used a lot of times so it 
# probably means something

# The tfidf instance has two parameters, first analyzer which is what the features will 
# be made out of, characters or words
# The ngram range is how many characters/words the feature will be made out of
# In this case I found this to be 3 and with characters that gives me the optimal 
# accuracy (~84%) (see post model visualizations below)
# This might not make sense as characters might not have inherent meaning when
# it comes to sentiment but in groups they can, the average word has a length of 5 
# characters, but this could be shortened down by the context of my dataset, twitter
# Twitter has character limits and like all social media slangs and abbriveations
# are really common, for example lol which is a shorter 3-character abrievieation of a 
# longer 3 word phrase 
tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3,3), strip_accents = 'ascii')

# I expiremented with changing the tfidf vectorizer parameters to tune my model
# For example I wanted to see what would happen if I didn't convert all my letters 
# to lowercase
# The default parameter is to convert everything to lowercase, so I changed it with 
# this line of code:tfidf = TfidfVectorizer(lowercase = False)
# This dropped by accuracy, but only by a very marginal amount, from 82.86% to 81.95%
# Thats a 0.91% drop so less than a percent
# Obviously I changed it back to converting all the words to
# lowercase for a higher accuracy
# It makes sense that not converting words to lowercase drops my accuracy becasue 
# it places a unneccesary high importance (higher idf) on words that start a sentence
# The only case where capatalized words actually are unique are with proper nouns but 
# it is hard for those to actually have meaning (like a name doesn't really have a
# positive or negative sentiment) so that caveat is irrelavent

# I also expiremented with deleting stop words or not to tune my model
# tfidf = TfidfVectorizer(stop_words = stop_words)
# This dropped my accuracy from 82.86% to 79.63% which is about 3.23% which is
#  decently substantial
# This seems counterintutive as removing stop words are supposed to increase 
# accuracy because usually they don't have an impact on the sentiment of words 
# so it allows the model to focus on the actual unique values
# But in the case of my model it makes sense that stop words could drop the 
# accuracy of my model 
# This is because I am using tfidf which already calculates the relavency of a word
# using the tf*idf algorithm
# This accounts for stop words as they will have a low idf which will cancel out 
# their high tf which will place the proper amount of importance on them
# This means if you delete the stop words it will mess with the model as the model
# already accounted for them
# This is one of the strengths of using tfidf 

# Additionally, I expiremented with what text the features should be made of
# For example, should they be made out of single words, or single characters?
# I tried testing out single characters and I had a strong feeling that my accuracy
# would be lower with chars than with single words, because it doesn't make sense for 
# single characters to affect the sentiment of a text, unless specific characters have 
# inherent positive/negative denotations, which are probably super obscure, words just 
# make more sense
# For example "hate" has a clear negative connotation but the letter "h" has no
# negative or positive connotation, it could be part of a larger positive word 
# like happy or a negative one like hate

# This is the code for using single characters
# The tfidf instance just takes in two parameters here the analyzer which 
# makes the features made out of characters
# and the ngram_range which decides how many characters to utilize in groups 
# in this case 1, 1 means only 1 character groups
# Code I used for the parameters 
#tfidf = TfidfVectorizer(analyzer = 'char', ngram_range = (1,1))
# This dropped my accuracy from 82.86% to 67.57% which is a 15.29% drop
# This is pretty substantial and again makes sense because specific 
# characters don't really have connotation 
# I then decided to change the n-grams to (2,2) which means it can also utilize
# bigrams which are two characters grouped together
# This boosted my accuracy from 67.57 to 79.45 
# This makes sense as groups of characters are alot closer to words than 
# I then tried this out with tri grams and my accuracy jumped to 84.03, which was 
# higher than using words
# This got me interested so I decided to create a chart to visualize accuracy with 
# different n-gram sizes (see below)

# Finally I expieremented with the strip_accents parameters that changes whether you 
# normalize the data or not using ascii and unicode character normalization 
# Including this boosted my accuracy by 0.04% which is very minor but still something
# This makes sense as normalizing the characters should boost the accuracy, as it places
# more equal weighting on accented and non-accented characters

# Intializes a label encoder for the y/target values or the "sentiment" column
encoder = LabelEncoder()

# Performs tfidf on the on the training data 
# fit_transform fits the data and transforms it to learn the idf for documents
X_train_tfidf = tfidf.fit_transform(X_train)
# Performs tfidf on the test data based on the idf learned from the training data
X_test_tfidf = tfidf.transform(X_test)

# Encodes the sentiment/train/y values
# Fit_transforms on the training data fits and normalizes the econding
Y_train_encoded = encoder.fit_transform(Y_train)
# transform simply normalizes the encoding
Y_test_encoded = encoder.transform(Y_test)

# Intializes the logistic regression model
logistic_regression = LogisticRegression()

# Fits the logistic regression model on the training data
logistic_regression.fit(X_train_tfidf, Y_train_encoded)

# Predicts with the model based off the X testing data
X_test_predicted = logistic_regression.predict(X_test_tfidf)

# Then calculates the accuracy by comparing the X testing data with their 
# respective y labels
lraccuracy = accuracy_score(Y_test_encoded, X_test_predicted)

# Prints out the accuracy of the regression
# Converts the lraccuray float to a string
# Prints the accuracy in a nice format as a percentage limited 
# to 3 decimal places (or 5 total digits) (unless accuracy is <10 or 100)
# In those cases would print 4 decimal places or 2 decimal places, respectively
print("Accuracy: " + str(100 * round(lraccuracy, 5)) + "%")

# Asks the user to input text and the model will predict the sentiment
input = input("Input Text To Predict Sentiment:")
# Performs tfidf on the input so the model can actually look at it
input_tfidf = tfidf.transform([input])
# Predicts on the transformed input data
input_predicted = logistic_regression.predict(input_tfidf)
# Unencodes the predicted label for the input using the inverse_transform func
# This is so that the user can actually understand what the predicted 
# label is, instead of just seeing a bunch of numbers
input_predicted_unencoded = encoder.inverse_transform(input_predicted)
# Prints out the label for the predicted sentiment on the user input
print("Predicted Sentiment For The Text You Inputed : " + str(input_predicted_unencoded))

# List that stores all the n-gram sizes and another list that will store
# their respective accuracy values 
ngrams_value_list_chars = []
accuracy_list_chars = []
# Loops thorugh 10 different n-gram sizes
for n in range(1, 11):
        # Creates a new tfidf instance for every n-gram size
        viz_tfidf = TfidfVectorizer(analyzer = 'char', ngram_range = (n,n))
        # Performs tfidf on the training data 
        # fit_transform fits the data and transforms it to learn the idf for documents
        X_train_tfidf = viz_tfidf.fit_transform(X_train)
        # Performs tfidf on the test data based on the idf learned from the training data
        X_test_tfidf = viz_tfidf.transform(X_test)
        # Fits the model for the new training data that is encoded based on the tfidf 
        # instance
        # Still uses the same y training data as this doesn't change 
        logistic_regression.fit(X_train_tfidf, Y_train_encoded)
        # Predicts with the model based off the new X testing data
        X_test_predicted = logistic_regression.predict(X_test_tfidf)
        # Then calculates the accuracy by comparing the X testing data with their 
        # respective Y labels
        # Once again using the old encoded test labels as the y values do not change 
        # for the tfidf n-grams sizes
        lraccuracy = accuracy_score(Y_test_encoded, X_test_predicted)
        # Appends the accuracy and the respective n values to the different lists
        # They will the corresponding with each other, so that the can be plotted
        # There is a times 100 to convert it to percent accuracy 
        accuracy_list_chars.append(lraccuracy*100)
        ngrams_value_list_chars.append(n)


# List that stores all the n-gram sizes and another list that will store 
# their respective accuracy values but this to calculate the accuracies for 
# words instead of characters
ngrams_value_list_words = []
accuracy_list_words = []
# Loops thorugh 10 different n-gram sizes
for n in range(1, 11):
        # Creates a new tfidf instance for every n-gram size
        viz_tfidf = TfidfVectorizer(analyzer = 'word', ngram_range = (n,n))
        # Performs tfidf on the training data 
        # fit_transform fits the data and transforms it to learn the idf for documents
        X_train_tfidf = viz_tfidf.fit_transform(X_train)
        # Performs tfidf on the test data based on the idf learned from the training data
        X_test_tfidf = viz_tfidf.transform(X_test)
        # Fits the model for the new training data that is encoded based on the tfidf 
        # instance
        # Still uses the same y training data as this doesn't change 
        logistic_regression.fit(X_train_tfidf, Y_train_encoded)
        # Predicts with the model based off the new X testing data
        X_test_predicted = logistic_regression.predict(X_test_tfidf)
        # Then calculates the accuracy by comparing the X testing data with their 
        # respective Y labels
        # Once again using the old encoded test labels as the y values do not change 
        # for the tfidf n-grams sizes
        lraccuracy = accuracy_score(Y_test_encoded, X_test_predicted)
        # Appends the accuracy and the respective n values to the different lists
        # They will the corresponding with each other, so that the can be plotted
        # There is a times 100 to convert it to percent accuracy 
        accuracy_list_words.append(lraccuracy*100)
        ngrams_value_list_words.append(n)


# Creates the graph that visualizes the ngrams vs. accuracy rate for 
# words and chars
# Set the size
plot.figure(figsize = (10, 10))
# Creates the plot with the ngram values as the x
# and the accuracy as the y
# Does it for both chars and words on the same graph 
plot.scatter(ngrams_value_list_chars, accuracy_list_chars)
plot.scatter(ngrams_value_list_words, accuracy_list_words)
# Sets the title and the labels for the axis
plot.title("N-gram Values For Words & Chars vs Accuracy Of Logistic Regression Model")
plot.xlabel("N-Gram Values")
plot.ylabel("Accuracy Of Logistic Regression Model (%)")
# Set the tick marks for the x values to go from 1 to 10 and go up by 1
plot.xticks(range(1, 11, 1))
# Shows the plot
plot.show()

# For the chars it looks like a reverse polynomial with a peak at trigrams 
# n = 3, with an accuracy of about 84%
# I have an explenation about why when I create an instance of my tfidf model 
        
# For the words graph it looks like the graph of a^-x with a being some
# constant
# In other words it starts at really high accuracy but drammatically drops, 
# the accuracy drops about 20% for 2 words and another 20% for three words
# then evens out and drops about 1% over the next 7ish n-grams, ending w/ 
# about a 40% accuracy rate for 10 ngrams
# This makes sense because words clumped together won't have signifigant sentiment 
# in this case
# This is because I am using the selected text column which already has a small # of 
# words so the more words you use the less features you have but also the impact on 
# sentiment actually just makes less and less sense
# for example a 5 word phrase won't have the same significane to sentiment as a singular
# word


