#import nltk
#nltk.download('all', halt_on_error=False)

from nltk.corpus import stopwords
sw = stopwords.words("english")

print len(sw)