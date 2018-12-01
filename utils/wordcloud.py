import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from os import mkdir, path

def createCloud(separated_tweets, intermediary_path=""):

    if intermediary_path!="":
        if not path.isdir('./wordcloud'+intermediary_path):
            mkdir('./wordcloud'+intermediary_path)
        
    for i, data in enumerate(separated_tweets):
        stopwords = STOPWORDS.union({"RT"})
        text = " ".join([j.text for j in data])
                    
        wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_font_size=130, max_words=200, width=1600, height=800).generate(text)
        plt.figure( figsize=(20,10), facecolor='k')
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.savefig("wordcloud{}/wordcloud{}.png".format(intermediary_path, i), bbox_inches='tight')
        plt.close('all')