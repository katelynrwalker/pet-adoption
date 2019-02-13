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

def add_breed_groups(pet_df):
    breeds = pd.read_csv('data_minus_images/breed_labels.csv')
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



def add_everything(df):
    df = load_sentiments(df)
    df = add_breed_groups(df)
    df = add_description_info(df)
    df = add_color_count(df)
    df = add_name_flag(df)
    
    return df