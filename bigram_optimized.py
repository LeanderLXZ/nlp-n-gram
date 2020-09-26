import re
from bigram import biGram

class biGramOptimized(biGram):
    
    def __init__(self):
        self.bigram_prob_en = self.get_bigram_prob(
            *self.get_bigram(self.read_file('EN.txt')))
        self.bigram_prob_fr = self.get_bigram_prob(
            *self.get_bigram(self.read_file('FR.txt')))
        self.bigram_prob_gr = self.get_bigram_prob(
            *self.get_bigram(self.read_file('GR.txt')))
    
    def get_bigram(self, words):
        # Bigrams list
        bigram_list = []
        # Counts for unique words
        word_counts = {}
        # Counts for bigrams
        bigram_counts = {}
        for i, word in enumerate(words):
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
            if i < len(words) - 1:
                bigram = (word, words[i+1])
                bigram_list.append((bigram))
                if bigram in bigram_counts:
                    bigram_counts[bigram] += 1
                else:
                    bigram_counts[bigram] = 1
        return word_counts, bigram_list, bigram_counts
    
    def predict(self, sentence):
        max_prob = 0
        language = None
        for bigram_prob, lang in [(self.bigram_prob_en, 'EN'),
                                  (self.bigram_prob_fr, 'FR'),
                                  (self.bigram_prob_gr, 'GR')]:
            prob_sentence = 1
            for i in range(len(sentence)):
                if i < len(sentence) - 1:
                    bigram_i = (sentence[i], sentence[i+1])
                    # Test if the bigram is unseen for training set
                    if bigram_prob.get(bigram_i):
                        prob_sentence *= \
                            bigram_prob[bigram_i]
                    else:
                        # Set a very low probability for unknown bigrams
                        prob_sentence *= 0.00001
            # Get the language with the highest probability
            if prob_sentence > max_prob:
                language = lang
                max_prob = prob_sentence
        return language
        
if __name__ == '__main__':
    biGramOptimized().main()
