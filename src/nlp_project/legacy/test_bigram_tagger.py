import nltk

from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents (categories = 'news')

bigramTagger = nltk.BigramTagger(brown_tagged_sents)

test_sent = 'Notice that the bigram tagger manages to tag every word in a sentence it saw during training, but does badly on an unseen sentence.'
tokens = nltk.word_tokenize(test_sent)

x = bigramTagger.tag(tokens)
print(x)