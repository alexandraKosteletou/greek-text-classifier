from sklearn.datasets import fetch_20newsgroups

#In order to get faster execution times for this first example we will work on a partial dataset
#with only 4 categories out of the 20 available in the dataset
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
x=twenty_train.target_names
print(x)

#print the first lines of the first loaded file
print("\n".join(twenty_train.data[0].split("\n")[:3]))

#extracting features from text files
#assign a fixed integer id to each word occuring in any document of the training set
#for each document #i, count the number of occurences of each word w and store it in x[i,j] as the value
#of feature #j where j is the index of word x in the dictionary

#tokenizing text with scikit learn
#text preprocessing, tokenizing and filtering of stopwords are all included in CountVectorizer which builds a dictionary of features
#and transforms the documents to feature vectors

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts  = count_vect.fit_transform(twenty_train.data)
y = X_train_counts.shape
print(y)
#gives number of rows and number of columns

#apply tf-idf
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
a= X_train_tf.shape
print(a)

