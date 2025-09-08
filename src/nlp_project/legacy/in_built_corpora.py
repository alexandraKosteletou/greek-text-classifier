from nltk.corpus import reuters

files = reuters.fileids()
print(files)

#words function on corpus object
words16097 = reuters.words(['test/16097'])
print(words16097)

#access specific number of words
words20= reuters.words(['test/16097']) [:20]
print(words20)

reuterGenres= reuters.categories()
print(reuterGenres)

for w in reuters.words(categories=['bop', 'coca']):
    print(w +' ', end='')
    if (w is '.'):
        print()