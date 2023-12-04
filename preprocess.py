import os
import re
import ast
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.tag import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('vader_lexicon')

def contains(x): 
    return int("shared" in str(x).lower())

def numBath(x): 
    if "half" in x.lower(): 
        return 0.5 
    else: 
        return float(x.split()[0]) 
    
def wordCount(x): 
    tokenizer = RegexpTokenizer(r'\w+') 
    tokens = tokenizer.tokenize(x)
    return len(tokens) 

def numAdjectives(text): 
    words = word_tokenize(text)

    # Tag each word with its part of speech
    pos_tags = pos_tag(words)

    # Count the adjectives
    adjective_count = sum(1 for word, tag in pos_tags if tag.startswith('JJ'))

    return adjective_count

def sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

def numBedroom(x): 
    if "bedroom" not in x: 
        return 1 
    else: 
        match = re.search(r'(\d+)\s*bedroom', x)
        return int(match.group(1)) 
    
def rating(x): 
    match_rating = re.search(r'★(\d+\.\d+)', x)
    try: 
        return float(match_rating.group(1)) if match_rating else 4.74
    except: 
        return 4.74     # average reported rating on airbnb

def preprocess(df:pd.DataFrame): 
        
    
    df['host_since'] = pd.to_datetime(df['host_since'])
    df["last_scraped"] = pd.to_datetime(df['last_scraped'])
    
    df["num_days_hosted"] = (df["last_scraped"] - df['host_since']).dt.days
    
    df = df.drop(["id", "scrape_id", "last_scraped", "calendar_last_scraped", "host_since"], axis=1)
    
    df['host_is_superhost'] = df['host_is_superhost'].fillna('unknown')
    one_hot_encoded = pd.get_dummies(df['host_is_superhost'], prefix='superhost')
    one_hot_encoded = one_hot_encoded.rename(columns={
        'superhost_t': 'superhost_true',
        'superhost_f': 'superhost_false',
        'superhost_unknown': 'superhost_unknown'
    }).astype(int)
    df = df.drop('host_is_superhost', axis=1).join(one_hot_encoded)

    for col in ["host_has_profile_pic", "host_identity_verified", "has_availability", "instant_bookable"]: 
        df.loc[df[col] == "f", col] = 0
        df.loc[df[col] == "t", col] = 1
    
    df["shared_bath"] = df["bathrooms_text"].apply(contains)
    df["num_baths"] = df["bathrooms_text"].apply(numBath)
    df = df.drop(["bathrooms_text"], axis=1)
    
    df = df.drop(["host_name"], axis=1)
    
    # one hot encode 
    dummies = df['host_verifications'].apply(lambda x: pd.Series({veri: 1 for veri in ast.literal_eval(x)}))
    dummies = dummies.fillna(0).astype(int)
    df = df.join(dummies)
    

    df["amenities"] = df["amenities"].apply(lambda x : len(ast.literal_eval(x))) 
    df = df.drop(["host_verifications", "amenities"], axis=1)
    
    # property_type and room_type 
    dummies = pd.get_dummies(df["room_type"])
    df = df.join(dummies) 
    df = df.drop(["property_type", "room_type"], axis=1)
    
    # Take 'name' and extract 'num_bedroom' (default=1) and 'rating' (default=mean(ratings))
    df["num_bedroom"] = df["name"].apply(numBedroom)
    df["rating"] = df["name"].apply(rating)
    
    # Take 'description' and extract the number of adjectives as a measure of "flashiness" of description
    
    # iterate through descriptions and calculate sentiment scores
    sentiments = df['description'].apply(sentiment)
    
    df["wordCount"] = df["description"].apply(wordCount)
    df["numAdjectives"] = df["description"].apply(numAdjectives)
    df["sentimentCompound"] = sentiments.apply(lambda x : x['compound'])
    df["sentimentPos"] = sentiments.apply(lambda x : x['pos'])
    df["sentimentNeg"] = sentiments.apply(lambda x : x['neg'])
    df["sentimentNeu"] = sentiments.apply(lambda x : x['neu'])
    df = df.drop(["description"], axis=1)
    
    one_hot = pd.get_dummies(df['neighbourhood_group_cleansed'])
    df = df.join(one_hot)
    
    # image 
    df = df.drop(["picture_url", "name", "neighbourhood_cleansed", "neighbourhood_group_cleansed"], axis=1)
    
    
    return df