import string
import re
from wordcloud import WordCloud
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import PunktSentenceTokenizer
from collections import Counter
import PyPDF4
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import ward, dendrogram

stopwords = nltk.corpus.stopwords.words('english')
# additional stopwords to be removed manually.
file = open('Corpus.txt', 'r')
moreStopwords = file.read().splitlines()
wn = nltk.WordNetLemmatizer()
tokenizer = PunktSentenceTokenizer()

data = PyPDF4.PdfFileReader(open('ReadyPlayerOne.pdf', 'rb'))
pageData = ''
for page in data.pages:
    pageData += page.extractText()


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokenize = re.split("\W+", text) # tokenizing based on words
    text = [wn.lemmatize(word) for word in tokenize if word not in stopwords]
    final = [word for word in text if word not in moreStopwords]
    return final


filter_data = clean_text(pageData)

top_100 = [(word, word_count) for word, word_count in Counter(filter_data).most_common(100)]
with open('top100_with_freq.txt', 'w') as fp:
    fp.write('\n'.join('%s %s' % x for x in top_100))
    fp.close()

most_common_words = [word for word, word_count in Counter(filter_data).most_common(100)]
freq = [word_count for word, word_count in Counter(filter_data).most_common(100)]
# plotting words & frequency
plt.bar(most_common_words, freq)
plt.xticks(rotation=90)
plt.show()

# Generating word cloud
data = TreebankWordDetokenizer().detokenize(filter_data)
#
cloud_mask = np.array(Image.open('twitter_mask.png'))
cloud = WordCloud(mask=cloud_mask, background_color='white', collocations=False, max_words=100).generate(data)
#
plt.imshow(cloud)
plt.axis('off')
plt.show()

# Feature Extraction
count_vectorizer = CountVectorizer(tokenizer=clean_text)
count_matrix = count_vectorizer .fit_transform(most_common_words)  # checking the words in MCW to find relevance
print(count_vectorizer .vocabulary_)
terms = count_vectorizer .get_feature_names()
dist = 1 - cosine_similarity(count_matrix)

# Clustering
km = KMeans(n_clusters=5, n_jobs=-1)
labels = km.fit_transform(count_matrix)
clusters = km.labels_.tolist()
X = count_matrix.todense()

# Displaying Cluster using Principal Component Analysis (PCA)
reduced_data = PCA(n_components=2).fit_transform(X)
xs, ys = reduced_data[:, 0], reduced_data[:, 1]

# Setting Up Cluster Colors
cluster_color = {
    0: '#FF0000', 1: '#00FF00', 2: '#0000FF', 3: '#FFFF00', 4: '#FFC0CB'
}
# Setting Up Cluster Labels
cluster_label = {
    0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3', 4: 'Cluster 4'
}

# create data frame that has the result of the PCA plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=most_common_words))

# group by cluster
groups = df.groupby('label')
fig, ax = plt.subplots(figsize=(17, 9))  # set size
# iterate through groups to layer the plot
# Using cluster_name and cluster_color with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_label[name],
            color=cluster_color[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    ax.tick_params(
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelleft=False)
ax.legend(numpoints=5)  # show legend with only 1 point

# add label in x,y position with the label as most_common_word
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)
plt.show()

linkage_matrix = ward(dist)  # define the linkage_matrix using ward clustering pre-computed distances
fig, ax = plt.subplots(figsize=(15, 20))  # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=most_common_words)
plt.tick_params(
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)

plt.tight_layout()  # show plot with tight layout
plt.show()
# uncomment below to save figure
# plt.savefig('ward_clusters.png', dpi=200)

