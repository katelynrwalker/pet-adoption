import pandas as pd
import json


def extract_sentiment(petID):
    '''
    Input:
        Target pet ID (JSON filename) to extract sentiments from.
    Output:
        Array of sentiments extracted from the JSONs.
    '''
    # Make sure this is run from the root
    try:
        with open('data_minus_images/train_sentiment/'+petID+'.json') as target:
            sentiment = json.load(target)
            magnitude, score = sentiment['documentSentiment'].values()
            language = sentiment['language']      
    except:
        magnitude, score, language = 0, 0, None
    return [magnitude, score, language]

def load_sentiments(pet_df):
    '''
    Input:
        Pet feature dataframe with unique PetIDs in a column.
    Output:
        Copy of the dataframe with sentiment metrics added as feature columns.
    '''
    output = pet_df.set_index('PetID')
    output['magnitude'] = 0
    output['score'] = 0
    output['language'] = None
    for pet in output.index.values:
        sentiments = extract_sentiment(pet)
        output.loc[pet,['magnitude','score','language']] = sentiments
    return output.reset_index()