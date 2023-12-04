import os
import re
import ast
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

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
    
    df.loc[df["host_is_superhost"] == "f", "host_is_superhost"] = -1 
    df.loc[df["host_is_superhost"] == "t", "host_is_superhost"] = 1 
    df["host_is_superhost"] = df["host_is_superhost"].fillna(0) 

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
    
    # dummies = df["amenities"].apply(lambda x : pd.Series({amen: 1 for amen in ast.literal_eval(x)}))
    # dummies = dummies.fillna(0).astype(int)
    # df = df.join(dummies)
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
    df["description"] = df["description"].apply(wordCount)
    
    # image 
    df = df.drop(["picture_url", "name", "neighbourhood_cleansed", "neighbourhood_group_cleansed"], axis=1)
    
    # df = df.to_numpy().astype(float)
    
    return df