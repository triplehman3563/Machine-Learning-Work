import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

documents = ["After shoot a combined 23 percent in Game 3, the three player were determined to play better than they had Sunday, and that meant squeezing in some extra repetitions. Some improvements were made -- Antetokounmpo finished with 25 points, 13 more than he had in Game 3 -- but it wasn't enough. The Milwaukee Bucks dropped Game 4 of the Eastern Conference finals to the Toronto Raptors 120-102, only the second time this season they have lost back-to-back game. We just came out flat in the third quarter, Antetokounmpo said. It's something we can get better at -- something we can fix. The Bucks' system is built to withstand an individual player's shoot slump, but Bledsoe is frustrated with just how long he has struggled to find the basket. According to SecondSpectrum tracking, Bledsoe is shoot just 27 percent on his jump shots this postseason, the worst among all player with at least 50 attempts. I tell him just forget about it, Middleton told ESPN. That's the only way you can play better, is if you stop thinking about it so much.",
             "Feel good, said Leonard, who finished with 19 points on 6-for-13 shoot in 34 minutes. Keep going, keep fighting. We have a chance to make history. Asked if the minutes from Game 3 caught up with him in Game 4, Leonard passed on answering. There's no excuses, he said. You're play basketball. We got a win tonight. For so much of these playoffs, the Raptors have been getting win because of Leonard's heroics. That was the case in both of the previous two game Toronto had play here at ScotiabankArena -- in Game 7 against the Sixers, in which he hit a classic game winner, and in Sunday's Game 3, when he play through those career-high 52 minutes. The privilege of having a transcendent superstar like Leonard isn't just the gift of the singular performance that win a game, though Leonard has done plenty of that over the past six weeks of the postseason.",
             "Antetokounmpo finished with 25 points, 10 rebounds and five assists. Khris Middleton scored a game-high 30 points, but Nikola Mirotic was the only other Bucks player with double-digit points. Milwaukee shoot just 31.4% on threes. “We’re going to have to finish better at the 3-point line or make more threes,” Budenholzer said. It is just the second time this season the Bucks lost two consecutive game, and even though they trailed Boston 1-0 in the conference semifinals, they now face their biggest test of the playoffs with a spot in the Finals distilled to a best-of-3. When the Raptors left Milwaukee, they had questions to answer and found them at home. Leaving Toronto, the Bucks have their own problems to solve. How to get more from starting point guard Eric Bledsoe and reserve guard Malcolm Brogdon? How to take some offensive pressure off Antetokounmpo and Middleton? And how to slow down Toronto’s offense?"]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words="english", max_features=80)
BOW = vectorizer.fit_transform(documents)
print('Bag of words in an alphabet order :')
print(BOW.toarray())
print('\n')
print('List of words in an order of occurrence time')
print(vectorizer.vocabulary_)
print('\n')

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
X = tfidf.fit_transform(BOW)
np.set_printoptions(precision=1)
print('TF-IDF of words in an alphabet order')
print(X.toarray())
print('\n')

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=2, random_state=0)
lda_output = lda.fit_transform(X)
print('Output of LDA')
print(lda_output)
print('\n')
lda_classification = np.argmax(lda_output, axis=1)
print('Topic allocation by LDA')
print(lda_classification)
print('\n')

n_top_words = 10
feature_names = vectorizer.get_feature_names()
HFW_list = []
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx))
    print(" ".join([feature_names[i] for i in topic.argsort() [:-n_top_words - 1:-1]]))

