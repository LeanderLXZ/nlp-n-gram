import re
from bigram import biGram

class biGramTuring(biGram):
    
    def __init__(self, unknown_ratio=0.1):
        # Inherit from biGram class
        super(biGramTuring, self).__init__(unknown_ratio)
    
    def get_bigram_prob(self, word_counts, bigram_list, bigram_counts):
        bigram_prob = {}
        n_words = len(word_counts)
        for bigram in bigram_list:
            # Add-one Smoothing here
            bigram_prob[bigram] = \
                bigram_counts[bigram] + 1 / (word_counts[bigram[0]] + n_words)
        return bigram_prob

if __name__ == '__main__':
    biGramTuring().main()
