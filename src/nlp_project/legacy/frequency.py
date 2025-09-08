import nltk
nltk.download('webtext')
from nltk.corpus import webtext
print(webtext.fileids())

fileid='singles.txt'
wbt_words=webtext.words(fileid)
fdist=nltk.FreqDist(wbt_words)
print('Count of the maximum appearing token"', fdist.max(), '": ', fdist[fdist.max()] )

print(fdist.N())

print(fdist.most_common(10))
print(fdist.tabulate())
print(fdist.plot(cumulative=True))