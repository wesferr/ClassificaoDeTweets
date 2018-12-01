#!/usr/bin/python3
import json

class Tweet:
    def __init__(self, text):
        self.matrix = []
        self.text = text

def read_tweets(path):
    tweets = []
    file = open(path, "r").read().split("\n")
    for i in range(len(file)-1):
        js = json.loads(file[i])
        tweets.append(Tweet(js['text']))
    return tweets
