from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

#define the category map ->it is a dictionary
category_map = {'talk.politics.misc': 'Politics', 'rec.autos': 'Autos', 'rec.sport.hockey' : 'Hockey',
                'sci.electronics' : 'Electronics', 'sci.med' : 'Medicine'}

#get the training data
training_set = fetch_20newsgroups(subset='train',
                                  categories= category_map.keys(),
                                  shuffle= True, random_state=5)

for idx, cat in enumerate(training_set.target_names):
    print(idx, cat)

#build training set->CountVectorizer is a module
#fit and transform are methods
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(training_set.data)
print("\nDimensions of training_set:", train_tc.shape)

#build the tf idf transformer
tfidf = TfidfTransformer()
train_tfidf  = tfidf.fit_transform(train_tc)

#sentences to test
input_data = [
    'You need to be careful with cars when you are driving on slippery roads',
    'A lot of devices can be operated wirelessly',
    'Players need to be careful when they are close to goal posts',
    'Political debates help us understand the perspectives of both sides'
]

#train a multinomial naive bayes classifier
classifier  = MultinomialNB().fit(train_tfidf, training_set.target)

#tranform input data by using count vectorizer
input_tc = count_vectorizer.transform(input_data)

#transform vectorized data using tfidf transformer
input_tfidf=tfidf.fit_transform(input_tc)

#predict the output catogories
predictions= classifier.predict(input_tfidf)
print(predictions)


# Print the outputs
for sent, category in zip(input_data, predictions):
    print('\nInput:', sent, '\nPredicted category:', \
            category_map[training_set.target_names[category]])