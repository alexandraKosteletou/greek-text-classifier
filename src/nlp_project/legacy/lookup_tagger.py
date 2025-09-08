import  nltk

y=()
sents=()

from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')


fd=()
baseline_tagger= ()
likely_tags = ()

#take a list of sentences, count the words and find the frequency distribution of words
fd = nltk.FreqDist(brown.words(categories='news'))

#store the frequency distribution of tags of each word of the tagged corpus
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))

#store the 100 most frequest words
most_freq_words = fd.most_common()[:100]

#store thhe most likely tag of each word of the 100 most frequent words
likely_tags = dict ((word, cfd[word].max()) for (word, _) in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model= likely_tags)
x = baseline_tagger.evaluate(brown_tagged_sents)
print(x)

