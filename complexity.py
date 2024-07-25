from nltk.wsd import lesk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pandas as pd
import ast
import nltk
lemmatizer = WordNetLemmatizer()

df = pd.read_csv("datasets/synonyms.csv", engine="c")
df.drop(columns=['Unnamed: 0'], inplace=True)


def get_sentence_words(sentence):

    # Tokenize the sentence into a list of words
    tokens = nltk.word_tokenize(sentence)

    # Perform POS tagging on the tokens
    pos_tags = nltk.pos_tag(tokens)

    # Extract the nouns, verbs, and adjectives from the sentence
    nouns = [word for (word, pos) in pos_tags if pos.startswith('N')]
    verbs = [word for (word, pos) in pos_tags if pos.startswith('V')]
    adjectives = [word for (word, pos) in pos_tags if pos.startswith('J')]
    adverbs = [word for (word, pos) in pos_tags if pos.startswith('R')]
    determiners = [word for (word, pos) in pos_tags if pos.startswith('D')]
    prepositions = [word for (word, pos) in pos_tags if pos.startswith('I')]

    # Print the nouns, verbs, and adjectives
    # print("Nouns:", nouns)
    # print("Verbs:", verbs)
    # print("Adjectives:", adjectives)
    # print("Aderbs:", adverbs)
    # print("Determiners:", determiners)
    # print("Prepositions:", prepositions)

    # Lemmatize each word in the sentence
    lemmatized_nouns = [lemmatizer.lemmatize(word) for word in nouns]
    lemmatized_verbs = [lemmatizer.lemmatize(word) for word in verbs]
    lemmatized_adjectives = [lemmatizer.lemmatize(word) for word in adjectives]
    lemmatized_adverbs = [lemmatizer.lemmatize(word) for word in adverbs]

    # Print the lemmatized words
    # print("Lemmatized nouns:",lemmatized_nouns)
    # print("Lemmatized verbs:",lemmatized_verbs)
    # print("Lemmatized adjectives:",lemmatized_adjectives)
    # print("Lemmatized adverbs:",lemmatized_adverbs)

    return lemmatized_verbs, adjectives, adverbs


def obtain_candidates(lemmatized_verbs, adjectives, adverbs):
    verbs_from_dataset = {}
    adjectives_from_dataset = {}
    adverbs_from_dataset = {}
    for synonyms in df['synonyms']:
        synonyms = ast.literal_eval(synonyms)
        for key, value in synonyms.items():
            if key in lemmatized_verbs:
                # min because lesser the times the word was used, more difficult it is
                verbs_from_dataset[key] = value
            elif (key in adjectives):
                

    # for i in range(len(df)):
    #     for j in df['synonyms'].iloc[i]:
    #         if (j in lemmatized_verbs):
    #             verbs_from_dataset[j] = df['synonyms'].iloc[i]
    #         elif (j in adjectives):
    #             # print("index: ", i, j)
    #             if j in adjectives_from_dataset:
    #                 for x in df['synonyms'].iloc[i]:
    #                     adjectives_from_dataset[j][x] = df['synonyms'].iloc[i][x]
    #             else:
    #                 adjectives_from_dataset[j] = df['synonyms'].iloc[i]
    #         elif (j in adverbs):
    #             adverbs_from_dataset[j] = df['synonyms'].iloc[i]
    return verbs_from_dataset, adjectives_from_dataset, adverbs_from_dataset


sentence_up = 'The quick fox lazily jumps over a conspicuous log very sleepily in a dazed state.'

VB, AD, AV = get_sentence_words(sentence_up)
print(df)
print(VB, AD, AV)
a, b, c = obtain_candidates(VB, AD, AV)
print(a, b, c)
