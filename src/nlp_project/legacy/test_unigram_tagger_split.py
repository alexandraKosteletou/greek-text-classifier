import nltk

from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories = 'news')

size = int(len(brown_tagged_sents) * 0.9)
#define the train set of the tagger
train_set = brown_tagged_sents[:size]

#unknown test sentence
test_sent = 'In the following code sample, we train a unigram tagger, use it to tag a sentence, then evaluate'
tokens = test_sent.split()

#train the tagger with train set
unigramTagger = nltk.UnigramTagger(train_set)

#test tagger on the unknown sentence
x= unigramTagger.tag(tokens)
print(x)
