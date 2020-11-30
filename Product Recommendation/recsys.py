import sys
import os
import random
import json

sys.path.append(os.path.abspath("../"))
#ROOT_DIR = os.path.abspath(os.curdir)
ROOT_DIR =  os.path.dirname(sys.modules['__main__'].__file__)
#from recsys import *
#from generic_preprocessing import *
#from IPython.display import HTML
#import pandas as pd
import warnings
warnings.filterwarnings("ignore")

## Importing required libraries
import pandas as pd ## For DataFrame operation
import numpy as np ## Numerical python for matrix operations
#from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler ## Preprocessing function
#import pandas_profiling ## For easy profiling of pandas DataFrame
#import missingno as msno ## Missing value co-occurance analysis

from scipy import sparse
from lightfm import LightFM
from sklearn.metrics.pairwise import cosine_similarity
import pickle

#import pandas as pd
#import numpy as np


def create_interaction_matrix(df,user_col, item_col, rating_col, norm= False, threshold = None):
    '''
    Function to create an interaction matrix dataframe from transactional type interactions
    Required Input -
        - df = Pandas DataFrame containing user-item interactions
        - user_col = column name containing user's identifier
        - item_col = column name containing item's identifier
        - rating col = column name containing user feedback on interaction with a given item
        - norm (optional) = True if a normalization of ratings is needed
        - threshold (required if norm = True) = value above which the rating is favorable
    Expected output - 
        - Pandas dataframe with user-item interactions ready to be fed in a recommendation algorithm
    '''
    interactions = df.groupby([user_col, item_col])[rating_col] \
            .sum().unstack().reset_index(). \
            fillna(0).set_index(user_col)
    if norm:
        interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
    return interactions

def load_model():
    fn = os.path.join(ROOT_DIR, 'finalized_model_V2.sav') 
    loaded_model = pickle.load(open(fn, 'rb'))
    return loaded_model

def product_dict():
    fn = os.path.join(ROOT_DIR, 'ProductDict_V2.sav') 
    ProductDict = pickle.load(open(fn, 'rb'))
    return ProductDict

def customer_dict():
    fn = os.path.join(ROOT_DIR, 'UserDict_V2.sav') 
    UserDict = pickle.load(open(fn, 'rb'))
    return UserDict

def item_matrix():
    fn = os.path.join(ROOT_DIR, 'ItemMatrix_V2.sav') 
    itemMatrix = pickle.load(open(fn, 'rb'))
    return itemMatrix

def inter_action():
    fn = os.path.join(ROOT_DIR, 'InterActions_V2.sav')
    InterActions = pickle.load(open(fn, 'rb'))
    return InterActions
def create_user_dict(interactions):
    '''
    Function to create a user dictionary based on their index and number in interaction dataset
    Required Input - 
        interactions - dataset create by create_interaction_matrix
    Expected Output -
        user_dict - Dictionary type output containing interaction_index as key and user_id as value
    '''
    user_id = list(interactions.index)
    user_dict = {}
    counter = 0 
    for i in user_id:
        user_dict[i] = counter
        counter += 1
    return user_dict

def random_item(num =10):
    #l = [801001366,801001497,801001479,801001425,801001378,801001386,704000206,801001533,801001868,801001441,704000204,704000140,801001658,801001521]
    #randomitem =  random.choice(l)
    #return randomitem
    s =['S.O.M.CORDYTIBET&BHUTAN','BALANSLIV','S.O.M.I-KARE','DERAEYDERAEY','S.O.M.S-BALANCE','BCCBCC','BONBACKBONBACK','DDNEEDDNEE','ACTIVISACTIVIS',
'AIKAIAIKAI','S.O.M.CMAX','CORDYCEPCORDYCEP','BENJAOILSECURMIN','REVIVEREVIVE','NUTRIPROFISHOIL','S.O.M.TIMECAPSULE','S.O.M.LINGZHISUN','ASTONASTON',
'S.O.M.MULTIPLUSCOLLAGEN','ULTIMATECOLLAGENULTIMATECOLLAGEN','REALELIXIRABALONECOLLAGEN']
    
    
    top_dict = {'S.O.M.CORDYTIBET&BHUTAN':'S.O.M.|CORDY TIBET & BHUTAN','BALANSLIV':'BALANS|LIV', 'S.O.M.I-KARE':'S.O.M.|I-KARE','DERAEYDERAEY':'DERAEY|DERAEY', 'S.O.M.S-BALANCE':'S.O.M.|S-BALANCE', 'BCCBCC':'BCC|BCC', 'BONBACKBONBACK':'BONBACK|BONBACK',
 'DDNEEDDNEE':'DD NEE|DD NEE', 'ACTIVISACTIVIS':'ACTIVIS|ACTIVIS', 'AIKAIAIKAI':'AI KAI|AI KAI', 'S.O.M.CMAX':'S.O.M.|C MAX','CORDYCEPCORDYCEP':'CORDYCEP|CORDYCEP', 'BENJAOILSECURMIN':'BENJA OIL|SECURMIN', 'REVIVEREVIVE':'REVIVE|REVIVE', 'NUTRIPROFISHOIL':'NUTRIPRO|FISH OIL', 'S.O.M.TIMECAPSULE':'S.O.M.|TIME CAPSULE', 'S.O.M.LINGZHISUN':'S.O.M.|LINGZHI SUN', 'ASTONASTON':'ASTON|ASTON', 'S.O.M.MULTIPLUSCOLLAGEN':'S.O.M.|MULTIPLUS COLLAGEN', 'ULTIMATECOLLAGENULTIMATECOLLAGEN':'ULTIMATE COLLAGEN|ULTIMATE COLLAGEN',
'REALELIXIRABALONECOLLAGEN':'REAL ELIXIR|ABALONE COLLAGEN'}
    sampled_list = list(random.sample(s,num))
    

    #print(sampled_list)
    x=1
    random_dic = {}
    for i in sampled_list:
        
        #print(i)
        random_dic.update({str(x): top_dict[i]})
        x = x + 1
    print(json.dumps(random_dic))


def create_item_dict(df,id_col,name_col):
    '''
    Function to create an item dictionary based on their item_id and item name
    Required Input - 
        - df = Pandas dataframe with Item information
        - id_col = Column name containing unique identifier for an item
        - name_col = Column name containing name of the item
    Expected Output -
        item_dict = Dictionary type output containing item_id as key and item_name as value
    '''
    item_dict ={}
    for i in range(df.shape[0]):
        item_dict[(df.loc[i,id_col])] = df.loc[i,name_col]
    return item_dict

def runMF(interactions, n_components=30, loss='warp', k=15, epoch=30,n_jobs = 4):
    '''
    Function to run matrix-factorization algorithm
    Required Input -
        - interactions = dataset create by create_interaction_matrix
        - n_components = number of embeddings you want to create to define Item and user
        - loss = loss function other options are logistic, brp
        - epoch = number of epochs to run 
        - n_jobs = number of cores used for execution 
    Expected Output  -
        Model - Trained model
    '''
    x = sparse.csr_matrix(interactions.values)
    model = LightFM(no_components= n_components, loss=loss,k=k)
    model.fit(x,epochs=epoch,num_threads = n_jobs)
    return model

def sample_recommendation_user(user_id,nrec_items = 10, show = True):
    '''
    Function to produce user recommendations
    Required Input - 
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
        - user_id = user ID for which we need to generate recommendation
        - user_dict = Dictionary type input containing interaction_index as key and user_id as value
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - threshold = value above which the rating is favorable in new interaction matrix
        - nrec_items = Number of output recommendation needed
    Expected Output - 
        - Prints list of items the given user has already bought
        - Prints list of N recommended items  which user hopefully will be interested in
    '''
    threshold = 5
    model = load_model()
    interactions = inter_action()
    user_dict = customer_dict()
    item_dict=product_dict()
    
    
    
    
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id]
    scores = pd.Series(model.predict(user_x,np.arange(n_items)))
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
    known_items = list(pd.Series(interactions.loc[user_id,:] \
                                 [interactions.loc[user_id,:] > threshold].index) \
								 .sort_values(ascending=False))
    
    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    if show == True:
        #print("Known Likes:")
        #counter = 1
        #for i in known_items:
        #    print(str(counter) + '- ' + i)
        #    counter+=1

        #print("\n Recommended Items:")
        counter = 1
        item_dic ={}
        for i in scores:
            #print(str(counter) + '- ' + str(i) )
            #print(str(i))
            item_dic.update({str(counter): str(i)})
            counter+=1
        print(json.dumps(item_dic))
    return return_score_list
    

def sample_recommendation_item(model,interactions,item_id,user_dict,item_dict,number_of_user):
    '''
    Funnction to produce a list of top N interested users for a given item
    Required Input -
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
        - item_id = item ID for which we need to generate recommended users
        - user_dict =  Dictionary type input containing interaction_index as key and user_id as value
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - number_of_user = Number of users needed as an output
    Expected Output -
        - user_list = List of recommended users 
    '''

    
    n_users, n_items = interactions.shape
    x = np.array(interactions.columns)
    scores = pd.Series(model.predict(np.arange(n_users), np.repeat(x.searchsorted(item_id),n_users)))
    user_list = list(interactions.index[scores.sort_values(ascending=False).head(number_of_user).index])
    return user_list 



def create_item_emdedding_distance_matrix(model,interactions):
    '''
    Function to create item-item distance embedding matrix
    Required Input -
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
    Expected Output -
        - item_emdedding_distance_matrix = Pandas dataframe containing cosine distance matrix b/w items
    '''
    df_item_norm_sparse = sparse.csr_matrix(model.item_embeddings)
    similarities = cosine_similarity(df_item_norm_sparse)
    item_emdedding_distance_matrix = pd.DataFrame(similarities)
    item_emdedding_distance_matrix.columns = interactions.columns
    item_emdedding_distance_matrix.index = interactions.columns
    return item_emdedding_distance_matrix

def item_item_recommendation( item_id, 
                             n_items = 10, show = True):
    '''
    Function to create item-item recommendation
    Required Input - 
        - item_emdedding_distance_matrix = Pandas dataframe containing cosine distance matrix b/w items
        - item_id  = item ID for which we need to generate recommended items
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - n_items = Number of items needed as an output
    Expected Output -
        - recommended_items = List of recommended items
    '''
    item_emdedding_distance_matrix = item_matrix()
    item_dict=product_dict()
    
    recommended_items = list(pd.Series(item_emdedding_distance_matrix.loc[item_id,:]. \
                                  sort_values(ascending = False).head(n_items+1). \
                                  index[1:n_items+1]))
    if show == True:
        #print("Item of interest :{0}".format(item_dict[item_id]))
        #print("Item similar to the above item:")
        product_dic ={}
        counter = 1
        for i in recommended_items:
            #print(str(counter) + '- ' +  item_dict[i])
            product_dic.update({str(counter): item_dict[i]})
            counter+=1
        print(json.dumps(product_dic))
    return recommended_items

    
if __name__ == "__main__":
     
    rectype = sys.argv[1]

   
    if rectype == "item":
        code = sys.argv[2]       
        num = sys.argv[3]
              
        #item_item_recommendation(float(code),int(num))
        try:
          item_item_recommendation(code,int(num))
        except:
           random_item()
          #item_item_recommendation(random_item(),int(num))
    elif rectype == "user":
        code = sys.argv[2]
        num = sys.argv[3]
        try:
            sample_recommendation_user(code,int(num))
        except:
            random_item() #item_item_recommendation(random_item(),int(num))
    elif rectype == "top":   
        num = sys.argv[2]
        random_item(int(num)) #item_item_recommendation(random_item(),int(num))
    
    
    #print(float(a))
    #print(a,b)

    
