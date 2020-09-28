import re
from bigram import biGram

class biGramOptimized(biGram):
    
    def __init__(self):
        super(biGramOptimized, self).__init__(0)
    
    def predict(self, sentence):
        max_prob = 0
        language = 'EN'
        for bigram_prob, lang in [(self.bigram_prob_en, 'EN'),
                                  (self.bigram_prob_fr, 'FR'),
                                  (self.bigram_prob_gr, 'GR')]:
            prob_sentence = 1
            for i in range(len(sentence)):
                if i < len(sentence) - 1:
                    bigram_i = (sentence[i], sentence[i+1])
                    # Test if the bigram is unseen for training set
                    if bigram_prob.get(bigram_i):
                        prob_sentence *= bigram_prob[bigram_i]
                    else:
                        # Set a very low probability for unknown bigrams
                        prob_sentence *= 0.0001
            # Get the language with the highest probability
            if prob_sentence > max_prob:
                language = lang
                max_prob = prob_sentence
        return language
        
if __name__ == '__main__':
    biGramOptimized().main()
