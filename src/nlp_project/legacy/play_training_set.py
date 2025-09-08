from sklearn.datasets import fetch_20newsgroups

#get the training data
training_set = fetch_20newsgroups(subset='train',
                                  categories= None,
                                  shuffle= True, random_state=5)

for idx, cat in enumerate(training_set.target_names):
    print(idx, cat)