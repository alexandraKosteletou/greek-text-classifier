import nltk
from nltk.tokenize import sent_tokenize , word_tokenize , wordpunct_tokenize

input_text = "Do you know word tokenization? Let's find out! It is quite interesting."

print("\nSentence tokenizer:")
print(nltk.tokenize.sent_tokenize(input_text))

print("\nWord tokenizer:")
print(nltk.tokenize.word_tokenize(input_text))


print("\nWordPunctuation tokenizer:")
print(nltk.tokenize.wordpunct_tokenize(input_text))