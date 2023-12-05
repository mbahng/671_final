import os
import re
import ast
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.tag import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('vader_lexicon')
# nltk.download('words')
# nltk.download('stopwords')

relevant_amenities = ['Stove', 'Iron', 'Smoking allowed', 'Microwave', 'Luggage dropoff allowed', 'Bed linens', 'First aid kit', 'Essentials', 'Refrigerator', 'Shower gel', 'Fire extinguisher', 'Dryer', 'Hot water', 'Oven', 'Free street parking', 'Long term stays allowed', 'Washer', 'Backyard', 'Shampoo', 'Kitchen', 'Wifi', 'Dishes and silverware', 'Coffee maker', 'Cooking basics', 'Hangers', 'Hair dryer', 'Heating', 'Carbon monoxide alarm', 'Air conditioning', 'Lock on bedroom door', 'Lockbox', 'Smoke alarm', 'Dedicated workspace', 'Free parking on premises', 'Self check-in', 'Security cameras on property', 'Clothing storage: closet and dresser', 'Shared patio or balcony', 'Private entrance', 'Rice maker', 'Paid washer – In building', 'Clothing storage', 'TV', 'Private living room', 'Keypad', 'Portable fans', 'Gas stove', 'Single level home', 'Ceiling fan', 'Laundromat nearby', 'Paid parking on premises', 'Elevator', 'Coffee maker: Keurig coffee machine', 'Patio or balcony', 'Gym', 'Shared hot tub', 'Exercise equipment', 'Washer –\xa0In building', 'Dryer –\xa0In unit', 'Coffee maker: drip coffee maker', 'HDTV', 'Shared pool - ', 'Shared BBQ grill', 'Whirlpool gas stove', 'Washer –\xa0In unit', 'Stainless steel single oven', 'Shared gym in building', 'Coffee maker: drip coffee maker, french press', 'Paid street parking off premises', 'Coffee maker: french press', 'Free parking garage on premises – 1 space', 'Resort access', 'Free parking on premises – 1 space', 'Paid parking garage off premises', 'Folding or convertible high chair - always at the listing', 'Paid parking garage on premises – 1 space', 'Clothing storage: walk-in closet and dresser', 'Free residential garage on premises – 1 space', 'Paid parking lot off premises', '50" HDTV with Amazon Prime Video, Disney+, Fire TV, HBO Max, Hulu, Netflix, Roku, standard cable', '65" HDTV', '50" HDTV', 'Shared hot tub - available all year, open specific hours', 'GE stainless steel gas stove', 'TV with Roku', 'Shared outdoor pool - available all year, open 24 hours, heated', 'Exercise equipment: yoga mat', 'Whirlpool refrigerator', 'Dove conditioner', 'HDTV with Netflix', '55" HDTV', 'Free parking garage on premises – 2 spaces', 'Fast wifi – 135 Mbps', 'GE gas stove', '55" HDTV with Roku', '55" HDTV with Amazon Prime Video', '50" TV with Roku', 'Whirlpool stainless steel oven', 'KitchenAid gas stove', '55" TV', 'EV charger - level 1', 'LG refrigerator', '55" HDTV with Netflix', 'Stainless steel stove', 'Other stainless steel gas stove', 'TV with standard cable, Netflix', 'Fast wifi – 57 Mbps', 'Private BBQ grill: charcoal', 'Exercise equipment: elliptical, stationary bike, treadmill', 'Dove shampoo', 'Private BBQ grill: electric', 'Free carport on premises – 1 space', 'Shared gym nearby', 'TV with Apple TV', '50" HDTV with standard cable', 'Samsung stainless steel gas stove', 'LG stainless steel gas stove', 'Fast wifi – 234 Mbps', 'Paid parking lot on premises', 'Bertazzoni gas stove', '55" HDTV with Amazon Prime Video, Hulu, Netflix', 'Samsung stainless steel oven', 'Fast wifi – 282 Mbps', '55" HDTV with standard cable', 'Coffee maker: Keurig coffee machine, pour-over coffee', 'Fast wifi – 101 Mbps', 'Soap body soap', 'Private gym nearby', 'Wifi – 45 Mbps', 'Fast wifi – 102 Mbps', 'LG stainless steel single oven', '50" HDTV with Netflix, standard cable', 'Fast wifi – 374 Mbps', 'Organic body soap', 'Exercise equipment: elliptical', 'Fast wifi – 340 Mbps', '65" HDTV with Roku', 'Shared BBQ grill: charcoal', 'Fast wifi – 465 Mbps', '58" HDTV', 'Outdoor kitchen with oven', 'Fast wifi – 104 Mbps', 'Coffee maker: french press, pour-over coffee', 'Exercise equipment: stationary bike, yoga mat', 'Changing table - available upon request', 'Head & Shoulders shampoo', 'Alexa Bluetooth sound system', 'Shared hot tub - available all year, open 24 hours', '52" HDTV', 'LG stainless steel oven', 'Window guards', 'Bathtub', 'Pets allowed', 'TV with standard cable', 'Room-darkening shades', 'Extra pillows and blankets', 'Dishwasher', 'Freezer', 'Dining table', 'Central heating', 'Toaster', 'Cleaning products', 'Board games', 'Fire pit', 'Mountain view', 'Body soap', 'Garden view', 'Private backyard – Fully fenced', 'Blender', 'Coffee', 'Outdoor furniture', 'BBQ grill', 'Smart lock', 'Books and reading material', 'Indoor fireplace: wood-burning', 'Baking sheet', 'Conditioner', 'Hot water kettle', 'Central air conditioning', 'Private backyard', 'Free dryer – In unit', 'Wine glasses', 'Ethernet connection', 'Trash compactor', 'Game console', 'Cleaning available during stay', 'Outdoor dining area', 'Shared sauna', 'Babysitter recommendations', 'Indoor fireplace', 'Private patio or balcony', 'Pocket wifi', 'Beach access', 'Waterfront', 'City skyline view', 'Drying rack for clothing', 'Beach essentials', 'Hammock', 'Crib', 'Pack ’n play/Travel crib', 'Bosch stainless steel gas stove', 'Bluetooth sound system', 'Private hot tub', 'Ocean view', 'Pool view', 'Fast wifi – 366 Mbps', 'Fast wifi – 317 Mbps', 'Outdoor kitchen with sink', 'Golf course view', 'Private hot tub - available all year', 'Private outdoor pool', 'Sea view', 'Kayak', 'High chair - always at the listing', 'Children’s books and toys for ages 0-2 years old, 2-5 years old, 5-10 years old, and 10+ years old', 'Private pool', 'Backyard - Fully fenced', 'Aesop shampoo', 'Aesop body soap', 'Bose Bluetooth sound system', 'Clothing storage: walk-in closet, closet, wardrobe, and dresser', 'Indoor fireplace: gas, wood-burning', 'Coffee maker: drip coffee maker, espresso machine', 'Private hot tub - available all year, open 24 hours', '75" HDTV with Amazon Prime Video, Apple TV, Disney+, HBO Max, Hulu, Netflix', 'Vineyard view', 'Exercise equipment: free weights, stationary bike, yoga mat', 'Thermador stainless steel oven', 'Various conditioner', 'Body wash body soap', 'Various shampoo', 'TV with standard cable, DVD player', 'Baby monitor - available upon request', 'Children’s books and toys for ages 2-5 years old, 5-10 years old, and 10+ years old', '55" HDTV with Apple TV', 'HDTV with standard cable', 'Fast wifi – 280 Mbps', 'Fast wifi – 83 Mbps', 'Private outdoor pool - available all year, open 24 hours, heated', 'Private outdoor pool - available all year, open specific hours, heated', 'Fast wifi – 89 Mbps', 'Private outdoor pool - ', 'Private outdoor pool - heated', '60" HDTV with premium cable', 'Children’s books and toys for ages 5-10 years old and 10+ years old', 'Exercise equipment: free weights, stationary bike', 'Fast wifi – 213 Mbps', 'Polk sound system with Bluetooth and aux', 'Viking stainless steel gas stove', 'Private hot tub - available all year, open specific hours', 'Outdoor shower', 'Pool', 'Hot tub', 'Outdoor kitchen', 'Private gym', 'Private gym in building', 'Private outdoor kitchen', 'Resort view', 'Exercise equipment: free weights, treadmill, yoga mat', 'Indoor fireplace: electric, gas', 'Private sauna', 'HDTV with premium cable', 'Viking stainless steel oven', 'Free driveway parking on premises – 6 spaces', 'SONOS Bluetooth sound system', 'Private beach access', 'Free parking garage on premises – 3 spaces', 'Outdoor kitchen with sink, oven', 'Fast wifi – 215 Mbps', '75" HDTV with standard cable', 'Sub zero refrigerator', 'Private outdoor pool - available all year, open specific hours, saltwater', 'HDTV with Netflix, Amazon Prime Video, standard cable', 'Exercise equipment: elliptical, free weights, yoga mat', 'Miele stainless steel gas stove', 'Kevin Murphy shampoo', 'Kevin Murphy conditioner', 'Miele refrigerator', 'Viking  stainless steel gas stove', 'Private outdoor pool - heated, saltwater', 'Coffee maker: drip coffee maker, espresso machine, Keurig coffee machine', '70" HDTV with premium cable', 'Private beach access – Beachfront', 'Subzero  refrigerator', 'Kitchen aid refrigerator', '75" HDTV with Amazon Prime Video, Apple TV, Chromecast, Disney+, Fire TV, HBO Max, Hulu, Netflix, Roku', 'Private pool - available all year, heated', 'Indoor fireplace: electric, gas, wood-burning', 'Hermes conditioner', 'Hermes shampoo', 'Hermes body soap', 'Indoor fireplace: ethanol, gas', 'Private outdoor pool - available all year, open 24 hours, heated, infinity', 'Miele oven', 'GE Monogram stainless steel oven', 'Free driveway parking on premises – 10 spaces', 'Game console: PS4 and Xbox 360', 'Private outdoor pool - available seasonally, saltwater', 'Private hot tub - available seasonally', '88" HDTV with Apple TV']    

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

    for col in ["host_has_profile_pic", "host_identity_verified", "has_availability", "instant_bookable", "host_is_superhost"]: 
        df.loc[df[col] == "f", col] = 0
        df.loc[df[col] == "t", col] = 1
    
    df["shared_bath"] = df["bathrooms_text"].apply(contains)
    df["num_baths"] = df["bathrooms_text"].apply(numBath)
    df = df.drop(["bathrooms_text"], axis=1)
    
    df = df.drop(["host_name", "host_id"], axis=1)
    
    # one hot encode 
    dummies = df['host_verifications'].apply(lambda x: pd.Series({veri: 1 for veri in ast.literal_eval(x)}))
    dummies = dummies.fillna(0).astype(int)
    df = df.join(dummies)
    
    df['amenities'] = df['amenities'].apply(ast.literal_eval)

    # Iterate over each element in relevant_amenities and create new columns
    for amenity in relevant_amenities:
        df[f"amen_{amenity}"] = df['amenities'].apply(lambda x: amenity in x).astype(int)
        
    # df = df.drop(["amenities"], axis=1)

    # df["amenities"] = df["amenities"].apply(lambda x : len(ast.literal_eval(x))) 
    df = df.drop(["host_verifications", "amenities"], axis=1)
    
    # property_type and room_type 
    dummies = pd.get_dummies(df["room_type"])
    df = df.join(dummies) 
    df = df.drop(["room_type"], axis=1)
    
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
    
    df = df.drop(["picture_url", "name", "neighbourhood_group_cleansed"], axis=1)
    
    df = df.drop(["property_type", "neighbourhood_cleansed"], axis=1)
    
    
    return df