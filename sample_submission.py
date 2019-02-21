import pandas as pd
import numpy as np
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


def extract_sentiment(petID, filepath):
    '''
    Input:
        Target pet ID (JSON filename) to extract sentiments from.
    Output:
        Array of sentiments extracted from the JSONs.
    '''
    # Make sure this is run from the root
    try:
        with open(filepath+petID+'.json') as target:
            sentiment = json.load(target)
            magnitude, score = sentiment['documentSentiment'].values()
            language = sentiment['language']
    except:
        magnitude, score, language = 0, 0, None
    return [magnitude, score, language]

def load_sentiments(pet_df, filepath):
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
        sentiments = extract_sentiment(pet, filepath)
        output.loc[pet,['magnitude','score','language']] = sentiments
    return output.reset_index()

def group_encoding(row, group_dict, group_ID):
    try:
        if row['Type']==2:
            row['BreedGroup']='CAT'
            row['BreedGroupID']=-1
        elif row['Breed1']==307 or row['Breed2']!=0:
            row['BreedGroup']='MIXED'
            row['BreedGroupID']=0
        else:
            group = group_dict[row['BreedName']]
            row['BreedGroup'] = group
            row['BreedGroupID'] = group_ID.loc[group].BreedID
    except:
        pass
    return row

def add_breed_groups(pet_df, filepath):
    breeds = pd.read_csv(filepath)
    encoding = pd.read_csv('breed_group_encoding.csv', header=None, encoding = "ISO-8859-1")
    encoding.set_index(0,inplace=True)
    group_dict = encoding.to_dict()[1]

    # Add breedname for group processing
    add_breeds = pd.merge(pet_df, breeds.drop(columns=['Type']),left_on='Breed1',right_on='BreedID').drop(columns=['BreedID'])
    # Preload with empty columns
    add_breeds['BreedGroup'] = 'MISC'
    add_breeds['BreedGroupID'] = 8

    group_ID=pd.DataFrame.from_dict({'MIXED':0,
                                  'HERDING':1,
                                  'HOUND':2,
                                  'TOY':3,
                                  'NON-SPORTING':4,
                                  'SPORTING':5,
                                  'TERRIER':6,
                                  'WORKING':7,
                                  'MISC':8,
                                  'FSS':9},orient='index',columns=['BreedID'])

    add_groups = add_breeds.apply(group_encoding,axis=1, args=[group_dict, group_ID])
    # Add purebred flag
    add_groups['purebred']=~((add_groups['BreedGroup']=='MIXED')|(add_groups['Breed2']!=0))*1

    return add_groups


def add_description_info(df):
    df['desc_len'] = df['Description'].apply(lambda x: len(str(x)))
    df['start_cap'] = df['Description'].apply(lambda x: 1*(str(x)[0]!=str(x)[0].lower()))
    return df


def add_color_count(df):
    df['num_colors']=1*(df['Color1']>0)+1*(df['Color2']>0)+1*(df['Color3']>0)
    return df

def add_name_flag(df):
    df['has_name']=~pd.isnull(df['Name'])*1
    return df

def add_everything(df, sentiment_filepath, breed_filepath):
    df = load_sentiments(df, sentiment_filepath)
    df = add_breed_groups(df, breed_filepath)
    df = add_description_info(df)
    df = add_color_count(df)
    df = add_name_flag(df)

    return df


def dummify(X, cat_features):
    for feat in cat_features:
#        print(feat)
        dummies = pd.get_dummies(X[feat], prefix=feat)
        dummies.drop(dummies.columns[-1], axis=1, inplace=True)
        X = X.drop(feat, axis=1).merge(dummies, left_index=True, right_index=True)
    return X


cat_features = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'State', 'language', 'BreedGroup']

data_test = pd.read_csv('data_minus_images/test/test.csv')
pet_df_test = add_everything(data_test, 'data_minus_images/test_sentiment/', 'data_minus_images/breed_labels.csv')
X_test = pet_df_test.drop(columns=['Name','RescuerID','Description','PetID', 'BreedName', 'BreedGroupID'])
X_test = dummify(X_test, cat_features)

data = pd.read_csv('data_minus_images/train.csv')
pet_df = add_everything(data, 'data_minus_images/train_sentiment/', 'data_minus_images/breed_labels.csv')

X = pet_df.drop(columns=['Name','RescuerID','Description','PetID','AdoptionSpeed', 'BreedName', 'BreedGroupID'])
y = pet_df['AdoptionSpeed'].astype('str')

X = dummify(X, cat_features)


gbc = GradientBoostingClassifier()
gbc.fit(X, y)
y_predict = gbc.predict(X_test)
y_predict.to_csv('sample_submission.csv', index=False)
