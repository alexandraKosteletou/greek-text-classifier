import nltk
from nltk.corpus import brown
x=()

brown_tagged_words = brown.tagged_words()
brown_tagged_sents= brown.tagged_sents()

#define variable for frequency of words in brown corpus
fd=nltk.FreqDist(brown.words())

#store keys-words of brown corpus
brown_words= fd.keys()

#define variable for the frequency of tags of each word in the tagged corpus
cfd = nltk.ConditionalFreqDist(brown.tagged_words())


#define model -> take the most frequent tag of each word
mystery_model= dict((word, cfd[word].max()) for word in brown_words)
x=mystery_model['can']
print(x)

myster_tagger = nltk.UnigramTagger(model=mystery_model)
text="The prime minister urged the public to take the jab when it is offered while scientists stressed the side-effects were extremely rare and the benefits of protection against coronavirus were great"
tokens=text.split()
y = myster_tagger.tag(tokens)
print(y)
#error in prime minister and public (public is given Adjective tag instead of Noun tag)

#evaluate tagger in brown corpus
a = myster_tagger.evaluate(brown_tagged_sents)
print(a)




