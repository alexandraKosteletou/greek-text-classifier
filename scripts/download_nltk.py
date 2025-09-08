import nltk
PACKAGES = [
    "punkt", "punkt_tab", "wordnet", "stopwords", "brown",
    "reuters", "webtext", "omw-1.4",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
]
for p in PACKAGES:
    try:
        nltk.download(p, quiet=True)
        print("Downloaded:", p)
    except Exception as e:
        print("Skip", p, "=>", e)
print("NLTK bootstrap complete.")
