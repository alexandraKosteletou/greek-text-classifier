#Naive Bayes

import pandas as pd
movies = pd.read_csv('movies.csv')
movies.head().round(2)

movies.describe().round(2)


pd.set_option('display.width', 75)
from  nltk.tokenize import casual_tokenize
bag_of_words =[]
from collections import Counter
for text in movies.text:
    bag_of_words.append(Counter(casual_tokenize(text)))
df_bows=pd.DataFrame.from_records(bag_of_words)
df_bows=df_bows.fillna(0).astype(int)

x=df_bows.shape


y=df_bows.head()


#find keywords that predict sentiment from natural language text
from sklearn.naive_bayes import MultinomialNB
nb= MultinomialNB()
nb= nb.fit(df_bows, movies.sentiment > 0)
movies['predicted_sentiment']=nb.predict(df_bows) * 8 - 4
movies['error']= (movies.predicted_sentiment - movies.sentiment).abs()
print(movies.error.mean())