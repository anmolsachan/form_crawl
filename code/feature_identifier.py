import csv

from nltk.stem import WordNetLemmatizer

from code.config import training_file_path

wnl = WordNetLemmatizer()



class FeatureIdentifier:

    def __init__(self):
        self.dic = self._load_csv(training_file_path)

    @staticmethod
    def _load_csv(path):
        dic = {}
        with open(path, "r")as csv_file:
            rows = csv.DictReader(csv_file)
            for row in rows:
                feature = row.pop("feature")
                dic[feature] = []
                for key, value in row.items():
                    if value == "yes":
                        dic[feature].append(key)
            return dic

    @staticmethod
    def is_plural(word):
        lemma = wnl.lemmatize(word, 'n')
        plural = True if word is not lemma else False
        return plural, lemma

    def singular_form(self, word):
        res = self.is_plural(word)
        return res[1]

    def feature_classifier(self, word):
        features = []
        for key in self.dic:
            for values in self.dic[key]:
                if word in values:
                    features.append(key)
        if not features:
            for key in self.dic.keys():
                if word in key:
                    features.append(key)

        return list(set(features))

    def process(self, terms):
        result = {}
        for term in terms:
            for word in term.split():
                singular_word = self.singular_form(word)
                result[word] = self.feature_classifier(singular_word)
        return result

if __name__ == "__main__":
    f = FeatureIdentifier()
    a = f.process(["actor", "actors", "director", "movie", "cast", "mpaa",
                   "gross"])
    print(a)

'''
Sample Output
    {'gross': ['us box office gross'], 'cast': ['actors', 'actresses'],
     'mpaa': ['mpaa rating'], 'movie': ['title type', 'title'],
     'director': ['directors'], 'actor': ['actors'], 'actors': ['actors']}
'''


