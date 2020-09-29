import re
from bigram import biGram

class triGramKneserNey(biGram):
    
    def __init__(self, unknown_ratio=0.8):
        
        # The ratio of unknown words
        self.UNKNOWN_RATIO = unknown_ratio
        
        # Read files
        self.word_list_en, words_en = \
            self.replace_unknown_words(self.read_file('EN.txt'))
        self.word_list_fr, words_fr = \
            self.replace_unknown_words(self.read_file('FR.txt'))
        self.word_list_gr, words_gr = \
            self.replace_unknown_words(self.read_file('GR.txt'))
       
        # Get unigrams of each language
        self.unigrams_en = self.get_unigrams(words_en)
        self.unigrams_fr = self.get_unigrams(words_fr)
        self.unigrams_gr = self.get_unigrams(words_gr)

        # Get bigrams of each language
        self.bigrams_en = self.get_bigrams(words_en)
        self.bigrams_fr = self.get_bigrams(words_fr)
        self.bigrams_gr = self.get_bigrams(words_gr)

        # Get trigrams of each language
        self.trigrams_en = self.get_trigrams(words_en)
        self.trigrams_fr = self.get_trigrams(words_fr)
        self.trigrams_gr = self.get_trigrams(words_gr)

        self.d = 0.25
    
    def text_normalization(self, text):
        words = []
        # Change text to lower cases
        text = text.lower()
        # Remove all non-alphanumeric chars
        text = re.sub(r'[^a-zA-Z0-9\.\?\!]', ' ', text)
        # Remove newlines/tabs, etc.
        text = re.sub(r'\s|\n\r\t', ' ', text)
        # Add beginning and end of sentence markers
        text = re.sub(r'\.|\?|\!', ' </s> <s>', text)
        # Split text to words
        for w in text.split():
            words.append(w)
        # Remove the '<s>' from the end
        if words[-1] == '<s>':
            words.pop()
        return words
            
    def read_file(self, file_name):
        with open(file_name, 'r') as f:
            return self.text_normalization(f.read())
    
    def replace_unknown_words(self, words):
        # Counts for words
        word_counts = {}
        for word in words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        
        # Get unknown words list
        n_unknown = int(len(word_counts) * self.UNKNOWN_RATIO)
        sorted_words = sorted(word_counts.items(), key=lambda x:x[1])
        unknown_words = [w[0] for w in sorted_words[:n_unknown]]

        # Replace the unknown words to <UNK>
        word_counts = {'<UNK>': 0}
        new_words = []
        for word in words:
            if word in unknown_words:
                word = '<UNK>'
            new_words.append(word)
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        word_list = word_counts.keys()
        return word_list, new_words
    
    def get_unigrams(self, words):
        unigrams = {}
        i = 0
        while i < len(words):
            if words[i] in unigrams:
                unigrams[words[i]] += 1
            else:
                unigrams[words[i]] = 1
            i += 1
        return unigrams
    
    def get_bigrams(self, words):
        bigrams = {}
        i = 0
        while i < len(words) - 1:
            if words[i] not in bigrams:
                bigrams[words[i]] = {}
            if words[i+1] not in bigrams[words[i]]:
                bigrams[words[i]][words[i+1]] = 1
            else:
                bigrams[words[i]][words[i+1]] += 1
            i += 1
        return bigrams
    
    def get_trigrams(self, words):
        trigrams = {}
        i = 0
        while i < len(words) - 2:
            if words[i] not in trigrams:
                trigrams[words[i]] = {}
            if words[i+1] not in trigrams[words[i]]:
                trigrams[words[i]][words[i+1]] = {}
            if words[i+2] not in trigrams[words[i]][words[i+1]]:
                trigrams[words[i]][words[i+1]][words[i+2]] = 1
            else:
                trigrams[words[i]][words[i+1]][words[i+2]] += 1
            i += 1
        return trigrams
    
    def kneser_ney_prob_bigram(self, w1, w2, unigrams, bigrams, trigrams):
        if w1 not in unigrams:
            unigrams[w1] = 1 - self.d
        if w2 not in unigrams:
            unigrams[w2] = 1 - self.d
        if w1 not in bigrams:
            bigrams[w1] = {}
        if w2 not in bigrams[w1]:
            bigrams[w1][w2] = 1 - self.d
        
        cont_count1 = 0
        cont_count2 = 0
        for w in trigrams:
            if w1 in trigrams[w] and w2 in trigrams[w][w1]:
                cont_count1 += 1
        for w in bigrams:
            if w1 in bigrams[w]:
                cont_count2 +=1

        prob_bigram = max(cont_count1 - self.d, 0) / float(cont_count2)
        prob_bigram += (self.d / unigrams[w1]) * (len(bigrams[w1])) * \
            (unigrams[w2] / sum(unigrams.values()))
        return prob_bigram
    
    def kneser_ney_prob_trigram(self, w1, w2, w3, unigrams, bigrams, trigrams):
        if w1 not in bigrams:
            bigrams[w1] = {}
        if w2 not in bigrams:
            bigrams[w2] = {}
        if w2 not in bigrams[w1]:
            bigrams[w1][w2] = 1 - self.d
        if w3 not in bigrams[w2]:
            bigrams[w2][w3] = 1 - self.d
        if w1 not in trigrams:
            trigrams[w1] = {}
        if w2 not in trigrams[w1]:
            trigrams[w1][w2] = {}
        if w3 not in trigrams[w1][w2]:
            trigrams[w1][w2][w3] = 1 - self.d

        prob_trigram = max(trigrams[w1][w2][w3] - self.d, 0) \
            / float(bigrams[w2][w3])
        prob_bigram = self.kneser_ney_prob_bigram(
            w2, w3, unigrams, bigrams, trigrams)
        prob_trigram += (self.d / bigrams[w1][w2]) * \
            (len(trigrams[w1][w2])) * prob_bigram
        return prob_trigram
    
    def predict(self, sentence):
        
        max_prob = 0
        language = 'EN'
        for unigrams, bigrams, trigrams, word_list, lang in \
                [(self.unigrams_en, self.bigrams_en,
                  self.trigrams_en, self.word_list_en, 'EN'),
                 (self.unigrams_fr, self.bigrams_fr,
                  self.trigrams_fr, self.word_list_fr, 'FR'),
                 (self.unigrams_gr, self.bigrams_gr,
                  self.trigrams_gr, self.word_list_gr, 'GR')]:
                    
            # Change unseen words in sentence to <UNK>
            sentence = ['<UNK>' if w not in word_list else w for w in sentence]
            
            # Calculate probability
            prob_sentence = 1
            for i in range(len(sentence)):
                if i < len(sentence) - 2:
                    trigram_i = (sentence[i], sentence[i+1], sentence[i+2])
                    trigram_prob = self.kneser_ney_prob_trigram(
                        *trigram_i, unigrams, bigrams, trigrams)
                    prob_sentence *= trigram_prob
                        
            # Get the language with the highest probability
            if prob_sentence > max_prob:
                language = lang
                max_prob = prob_sentence
        return language

if __name__ == '__main__':
    triGramKneserNey().main()
