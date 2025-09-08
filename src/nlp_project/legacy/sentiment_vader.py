from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sa= SentimentIntensityAnalyzer()
sa.lexicon


x=[(tok, score) for tok, score in sa.lexicon.items() if " " in tok]
print(x)

y=sa.polarity_scores(text = "Python us very readable and it's great for NLP")
print(y)

corpus= ["Absolutely perfect! Love it! :-) :-) :-)", "Horrible! Completely useless. :(", "It was OK."]
for doc in corpus:
    scores=sa.polarity_scores(doc)
    print('{:+}: {}'.format(scores['compound'],doc))
