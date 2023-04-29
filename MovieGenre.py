import pyspark
import findspark
import pandas as pd
from pyspark.sql import *
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.sql import SQLContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import split, regexp_replace
from pyspark.sql.functions import col, udf, lit, explode
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import when, concat, explode,concat_ws
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, Word2Vec
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.sql.types import StructType, StructField, LongType, IntegerType

sc = pyspark.SparkContext()
spark = SparkSession.builder \
        .master("local") \
        .appName("myapp") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

sqlContext = SQLContext(sc)
sql = SQLContext(spark)

# Train Data
traindata = pd.read_csv("train.csv")
train_data = sql.createDataFrame(traindata)
train_data.show(2)

# Test Data
testdata = pd.read_csv("test.csv")
test_data = sql.createDataFrame(testdata)
test_data.show(2)

# Mapping Data
mappingdata = pd.read_csv("mapping.csv")
mapping_data = sql.createDataFrame(mappingdata)
mapping_data.show(2)

mapping_data = (mapping_data.withColumnRenamed("Unnamed: 0","ID").withColumnRenamed("0","Genre")).collect()
map_dictionary = {}
for i in range(0,20):
    map_dictionary[i] = mapping_data[i]['Genre'] 

# Tokenizer

rt = RegexTokenizer(inputCol = "plot", outputCol = "tokenized_terms", pattern = "\\W")
train_df = rt.transform(train_data)
test_df = rt.transform(test_data)

swr = StopWordsRemover(inputCol = "tokenized_terms", outputCol = "updated_terms")
train_df = swr.transform(train_df)
test_df = swr.transform(test_df)

train_df = train_df.drop('tokenized_terms')
test_df = test_df.drop('tokenized_terms')

train_df1 = train_df
train_df2 = train_df
train_df3 = train_df
test_df1 = test_df
test_df2 = test_df
test_df3 = test_df

mapping_df = pd.read_csv("mapping.csv", index_col = 0)

# Fitting the Count Vectorizer over our train and test data
cv = CountVectorizer(inputCol = "updated_terms", outputCol = "features", vocabSize = 9000, minDF = 8)
model = cv.fit(train_df1)
train_df1 = model.transform(train_df1)
train_df1.show(n = 5)

model = cv.fit(test_df1)
test_df1 = model.transform(test_df1)
test_df1.show(n = 5)

# Start of Random Forest Implementation
str_ind = StringIndexer(inputCol = "genre", outputCol = "multiLabels")
model = str_ind.fit(train_df1)
train_df1 = model.transform(train_df1)
train_df1.show(n=1)
u_labels = model.labels

rfc_model = RandomForestClassifier(labelCol="multiLabels", \
                            featuresCol="features", \
                            numTrees = 10, \
                            maxDepth = 4, \
                            maxBins = 10)
# Train model with Training Data
train_Model = rfc_model.fit(train_df1)

test_pred1 = train_Model.transform(test_df1)
ind_to_str = IndexToString(inputCol="prediction", outputCol="ClassLabel", labels = u_labels)
test_result = ind_to_str.transform(test_pred1)
test_result.columns

result_values = test_result.select('movie_id','ClassLabel').collect()
result = [['movie_id','predictions']]
for idx,i in enumerate(result_values):
#     print(''.join(str([1 if x[0]==i else 0 for x in mapping_df.values])))
    cats = str(i['ClassLabel']).split(',')
    cats = [x.strip("[]\'' ") for x in cats]
    val = ''.join(map(str,[str(1)+' ' if x[0] in cats else str(0)+' ' for x in mapping_df.values]))
    result.append([str(i['movie_id']),val.strip(' ')])
print(result[5983])

result = pd.DataFrame(result)
result.to_csv('rf1.csv',index = False, header = False) #Saving the csv file to our local folder

hashTF_init = HashingTF(inputCol = "updated_terms", outputCol = "rawfeatures", numFeatures = 10000)
hashtrain_transform = hashTF_init.transform(train_df2)
idftrain = IDF(inputCol = "rawfeatures", outputCol = "features")
idftrain_fit = idftrain.fit(hashtrain_transform)
train_df2 = idftrain_fit.transform(hashtrain_transform)

#hashTF_init = HashingTF(inputCol="updatedterms", outputCol="rawfeatures", numFeatures=9000)
hashtest_transform = hashTF_init.transform(test_df2)
idftest = IDF(inputCol = "rawfeatures", outputCol = "features")
idftest_fit = idftest.fit(hashtest_transform)
test_df2 = idftest_fit.transform(hashtest_transform)

# Start of Random Forest Implementation
str_ind = StringIndexer(inputCol = "genre", outputCol = "multiLabels")
model = str_ind.fit(train_df2)
train_df2 = model.transform(train_df2)
train_df2.show(n=1)
u_labels = model.labels

rfc_model = RandomForestClassifier(labelCol="multiLabels", \
                            featuresCol="features", \
                            numTrees = 10, \
                            maxDepth = 4, \
                            maxBins = 10)
# Train model with Training Data
train_Model = rfc_model.fit(train_df2)

test_pred2 = train_Model.transform(test_df2)
ind_to_str = IndexToString(inputCol="prediction", outputCol="ClassLabel", labels = u_labels)
test_result = ind_to_str.transform(test_pred2)
test_result.columns

result_values = test_result.select('movie_id','ClassLabel').collect()
result = [['movie_id','predictions']]
for idx,i in enumerate(result_values):
#     print(''.join(str([1 if x[0]==i else 0 for x in mapping_df.values])))
    cats = str(i['ClassLabel']).split(',')
    cats = [x.strip("[]\'' ") for x in cats]
    val = ''.join(map(str,[str(1)+' ' if x[0] in cats else str(0)+' ' for x in mapping_df.values]))
    result.append([str(i['movie_id']),val.strip(' ')])
print(result[5983])

result = pd.DataFrame(result)
result.to_csv('rf2.csv',index = False,header = False)

word2vec = Word2Vec(vectorSize = 300, minCount = 10, inputCol = 'updated_terms', outputCol = 'features')
word2vec_transform = word2vec.fit(train_df3)
train_df3 = word2vec_transform.transform(train_df3)

word2vec = Word2Vec(vectorSize = 250, minCount = 5, inputCol = 'updated_terms', outputCol = 'vectors')
word2vec_transform = word2vec.fit(test_df3)
test_df3 = word2vec_transform.transform(test_df3)

# Start of Random Forest Implementation
str_ind = StringIndexer(inputCol = "genre", outputCol = "multiLabels")
model = str_ind.fit(train_df3)
train_df3 = model.transform(train_df3)
train_df3.show(n=1)
u_labels = model.labels

rfc_model = RandomForestClassifier(labelCol="multiLabels", \
                            featuresCol="features", \
                            numTrees = 10, \
                            maxDepth = 4, \
                            maxBins = 10)
# Train model with Training Data
train_Model = rfc_model.fit(train_df3)

test_pred3 = train_Model.transform(train_df3)
ind_to_str = IndexToString(inputCol="prediction", outputCol="ClassLabel", labels = u_labels)
test_result = ind_to_str.transform(test_pred3)
test_result.columns

result_values = test_result.select('movie_id','ClassLabel').collect()
result = [['movie_id','predictions']]
for idx,i in enumerate(result_values):
#     print(''.join(str([1 if x[0]==i else 0 for x in mapping_df.values])))
    cats = str(i['ClassLabel']).split(',')
    cats = [x.strip("[]\'' ") for x in cats]
    val = ''.join(map(str,[str(1)+' ' if x[0] in cats else str(0)+' ' for x in mapping_df.values]))
    result.append([str(i['movie_id']),val.strip(' ')])
print(result[5983])

result = pd.DataFrame(result)
result.to_csv('rf3.csv',index = False,header = False)