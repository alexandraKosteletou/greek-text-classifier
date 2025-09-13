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

# Headless-safe plotting with fallback
try:
     import matplotlib
     matplotlib.use("Agg")
     import matplotlib.pyplot as plt  # noqa: F401
     fdist.plot(cumulative=True)
except Exception as e:
     print("Skipping plot (no matplotlib/headless CI):", e)
     print("Top 20:", fdist.most_common(20))