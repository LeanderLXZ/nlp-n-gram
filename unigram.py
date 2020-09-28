import re

class uniGram(object):
    
    def __init__(self, unknown_ratio=0.001):
        
        # The ratio of unknown words
        self.UNKNOWN_RATIO = unknown_ratio
        
        # Read files
        self.word_list_en, words_en, word_counts_en = \
            self.replace_unknown_words(self.read_file('EN.txt'))
        self.word_list_fr, words_fr, word_counts_fr = \
            self.replace_unknown_words(self.read_file('FR.txt'))
        self.word_list_gr, words_gr, word_counts_gr = \
            self.replace_unknown_words(self.read_file('GR.txt'))
       
        # Get unique probabilities of each language
        self.unique_prob_en = \
            self.get_unique_prob(self.word_list_en, word_counts_en)
        self.unique_prob_fr = \
            self.get_unique_prob(self.word_list_fr, word_counts_fr)
        self.unique_prob_gr = \
            self.get_unique_prob(self.word_list_gr, word_counts_gr)
    
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
        return word_list, new_words, word_counts
    
    def get_unique_prob(self, word_list, word_counts):
        unique_prob = {}
        for word in word_list:
            unique_prob[word] = word_counts[word] / len(word_counts)
        return unique_prob
    
    def predict(self, sentence):
        
        max_prob = 0
        language = 'EN'
        for unigram_prob, word_list, lang in \
                [(self.unique_prob_en, self.word_list_en, 'EN'),
                 (self.unique_prob_fr, self.word_list_fr, 'FR'),
                 (self.unique_prob_gr, self.word_list_gr, 'GR')]:
                    
            # Change unseen words in sentence to <UNK>
            sentence = ['<UNK>' if w not in word_list else w for w in sentence]
            
            # Calculate probability
            prob_sentence = 1
            for i, unigram in enumerate(sentence):
                if i < len(sentence):
                    # Test if the unigram is unseen for training set
                    if unigram_prob.get(unigram):
                        prob_sentence *= unigram_prob[unigram]
                    else:
                        # If bigram is unknown
                        prob_sentence *= 0
                        
            # Get the language with the highest probability
            if prob_sentence > max_prob:
                language = lang
                max_prob = prob_sentence
        return language
        
    def main(self):
        
        with open('LangID.test.txt', 'r') as f:
            sentences = [re.sub(r'^\d+\. ', '', l)[:-1] for l in f.readlines()]
            
        with open('LangID.gold.txt', 'r') as f:
            languages = [l[-3:-1] for l in f.readlines()[1:]]
        
        accuracy = []
        for x, y in zip(sentences, languages):
            pred_y = self.predict(self.text_normalization(x))
            accuracy.append(pred_y == y)
        accuracy = sum(accuracy) / len(accuracy)
        
        print('Accuracy: {:.2f}%'.format(accuracy * 100))

if __name__ == '__main__':
    uniGram().main()
