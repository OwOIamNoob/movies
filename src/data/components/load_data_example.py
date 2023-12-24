import pandas
import numpy as np
import cv2
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
import json
from vocab import Vocab
# Read in the dataset, and do a little preprocessing,
# mostly to set the column datatypes.
vocab = Vocab()
users = pandas.read_csv('./users.dat', sep='::',
                        engine='python',
                        names=['userid', 'gender', 'age', 'occupation', 'zip']).set_index('userid')
user_ratings = pandas.read_csv('./user_rating.csv')
movies_train = pandas.read_csv('./movies_train.dat', engine='python',
                         sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')
movies_test = pandas.read_csv('./movies_test.dat', engine='python',
                         sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')                         
movies_train['genre'] = movies_train.genre.str.split('|')
movies_test['genre'] = movies_test.genre.str.split('|')

ages = sorted(np.unique(np.array(user_ratings['age'].tolist())))
user_ratings['gender'].replace(['F', 'M'], [0, 1], inplace=True)
user_ratings['age'].replace(ages, range(len(ages)), inplace=True)
user_ratings.drop(['userid', 'zip', 'timestamp'], axis='columns')
user_ratings['movieid'] = user_ratings['movieid'].astype('category')
img_dir =  "/work/hpc/potato/movies/data/movies/dataset/ml1m-images/{}.jpg"

test_dataset = dict()
i = 0
max_ratings = 0
for movieid in movies_test.index.to_numpy().tolist():
    test_dataset[movieid] = dict()
    frame = movies_test.loc[[movieid]] 
    test_dataset[movieid]['title'] = frame['title'].tolist()[0]
    tokens = vocab.tokenize(frame['title'].tolist()[0])
    test_dataset[movieid]['genre'] = frame['genre'].tolist()[0]
    try:
        ratings = sorted(user_ratings[user_ratings.movieid == movieid].drop(['userid', 'zip', 'timestamp', 'movieid'], axis='columns').to_numpy().tolist(), key=lambda a: (a[3], a[1], a[2], a[0]))
        max_ratings = max(len(ratings), max_ratings)
        # ratings_str = "\n".join([",".join([str(rate) for rate in rating]) for rating in ratings])  
    except:
        ratings = []
    test_dataset[movieid]['ratings'] = ratings
    img = cv2.imread(img_dir.format(movieid))
    if img is None:
        test_dataset[movieid]['image'] = False
        i += 1
    else:
        test_dataset[movieid]['image'] = True 

print(i)
print(max_ratings)
# xml = dicttoxml(train_dataset)
# dom = parseString(xml).toprettyxml()
# # dom_decoded = dom.decode()
# xml_file = open("/work/hpc/potato/movies/data/movies/dataset/train.xml", "w")
# xml_file.write(dom)
# xml_file.close()

with open("/work/hpc/potato/movies/data/movies/test.json", "w") as f:
    json.dump(test_dataset, f)
    f.close()

train_dataset = dict()
i = 0
max_ratings = 0
for movieid in movies_train.index.to_numpy().tolist():
    train_dataset[movieid] = dict()
    frame = movies_train.loc[[movieid]] 
    tokens = vocab.tokenize(frame['title'].tolist()[0])
    train_dataset[movieid]['title'] = frame['title'].tolist()[0]
    train_dataset[movieid]['genre'] = frame['genre'].tolist()[0]
    try:
        ratings = sorted(user_ratings[user_ratings.movieid == movieid].drop(['userid', 'zip', 'timestamp', 'movieid'], axis='columns').to_numpy().tolist(), key=lambda a: (a[3], a[1], a[2], a[0]))
        max_ratings = max(len(ratings), max_ratings)
        # ratings_str = "\n".join([",".join([str(rate) for rate in rating]) for rating in ratings])  
    except:
        ratings = []
    train_dataset[movieid]['ratings'] = ratings
    img = cv2.imread(img_dir.format(movieid))
    if img is None:
        train_dataset[movieid]['image'] = False
        i += 1
    else:
        train_dataset[movieid]['image'] = True 

print(i)
print(max_ratings)
words = "\n".join(list(vocab.vocab))
with open("/work/hpc/potato/movies/data/movies/words.txt", "w") as f:
    f.write(words)
    f.close()
with open("/work/hpc/potato/movies/data/movies/train.json", "w") as f:
    json.dump(train_dataset, f)
    f.close()