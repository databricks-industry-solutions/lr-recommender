# Databricks notebook source
# MAGIC %md The purpose of this notebook is to train the classifier recommender. You may find this notebook at https://github.com/databricks-industry-solutions/lr-recommender.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC In this notebook, we'll assemble the labels and features required to train a binary classifier that will serve as the basis of our recommender system.  Our labels will be derived from the implied ratings calculated in notebook *01* and our features will be retrieved from the feature store tables populated in notebook *02*.  The model we train will be persisted for deployment in the final notebook of this solution accelerator.

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn

import databricks.feature_store as feature_store
from databricks.feature_store import FeatureStoreClient

from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder 
from sklearn.pipeline import Pipeline as Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from pyspark.ml.evaluation import RankingEvaluator

import mlflow
from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval

import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md ##Step 1: Get Labels
# MAGIC
# MAGIC The implied ratings calculated in notebook *01* are indicators of a user's preference for a product.  For this recommender, we need to convert that numerical score to a 0 to indicate it is not preferred and a 1 to indicate that it is preferred.  The easiest way to do this is simply to define a threshold value below and above which we will assign these values.
# MAGIC
# MAGIC To help us determine where this threshold might be, we can generate a histogram:

# COMMAND ----------

# DBTITLE 1,Review Distribution of Ratings
display(
  spark
    .table('user_product_purchases')
    .select('rating')
    .sample(0.01) # take a 1% random sample 
  )

# COMMAND ----------

# MAGIC %md From the histogram, we can see our ratings values are highly skewed towards the lower end of the range of potential values.  So while our ratings range from 0.0 to 1.0, we cannot set a simple threshold of 0.5 unless we want our recommender to be biased towards those values with extremely high preferences.
# MAGIC
# MAGIC With that in mind, we might ask, what is the median value of our ratings?  This value, also known as the 50th-percentiel value, should evenly partition our value between those with lower preferences and those with higher preferences:

# COMMAND ----------

# DBTITLE 1,Calculate 50th Percentile (Median) Value
display(
  spark
    .table('user_product_purchases')
    .select('rating')
    .groupBy()
      .agg(
        fn.expr("percentile_cont(array(0.5)) within group (order by rating)").alias('median_rating') # median value is value at 0.50
        )
  )

# COMMAND ----------

# MAGIC %md The median value is quite low.  Still, this provides us a relatively even split.  We say relatively even because the *percentile_cont* function (and it's alias function for the 50th-percentile, *median*,) employ interpolation on an assumed continuous distribution.  In short, we would expect a near even but not perfectly even split on a set of values: 

# COMMAND ----------

# DBTITLE 1,Define Labels using Median
# MAGIC %sql
# MAGIC
# MAGIC CREATE OR REPLACE VIEW user_product_labels
# MAGIC AS
# MAGIC   SELECT
# MAGIC     a.user_id,
# MAGIC     a.product_id,
# MAGIC     a.rating,
# MAGIC     case 
# MAGIC       when a.rating < b.med then 0
# MAGIC       else 1
# MAGIC       end as label
# MAGIC   FROM user_product_purchases as a
# MAGIC   CROSS JOIN (
# MAGIC     SELECT
# MAGIC      median(rating) as med
# MAGIC     FROM user_product_purchases
# MAGIC     ) as b

# COMMAND ----------

# DBTITLE 1,Examine Label Distribution
display(
  spark
    .table('user_product_labels')
    .groupBy('label')
      .agg(fn.count('*').alias('instances'))
  )

# COMMAND ----------

# MAGIC %md With our data labeled, we can load these labels along with the *user_id* and *product_id* fields associated with them.  These additional fields will provide us the means to lookup features for each label: 

# COMMAND ----------

# DBTITLE 1,Assemble Labels & Feature Keys
labels = (
  spark
    .table('user_product_labels')
    .select('user_id','product_id','label','rating')
  )

display(labels)

# COMMAND ----------

# MAGIC %md It's important to note here that we are not considering user-product combinations that have not been purchased. The lack of purchase might indicate a lack of preference, but it might also imply the customer simply hasn't had the opportunity or awareness to make a purchase of a given product from us.  By limiting our data to just the purchased user-product combinations, we better leave open the opportunity that our recommender can suggest new products to users based on a user and product's more general characteristics.  That's not to say there may be scenarios where you might need to rethink this approach, but this seems to make sense for the general purpose we are pursuing here.

# COMMAND ----------

# MAGIC %md ##Step 2: Get Features
# MAGIC
# MAGIC We will now lookup the features to associate with each labeled instance.  The instructions defining how *user_id* and *product_id* values will be used to retrieve features from the feature store are defined as [*FeatureLookups*](https://docs.databricks.com/dev-tools/api/python/latest/feature-store/entities/feature_lookup.html):

# COMMAND ----------

# DBTITLE 1,Define Feature Lookups
feature_lookups = [
  # user features
  feature_store.FeatureLookup(
    table_name = f"{config['database']}.user_features",
    lookup_key = ['user_id'],
    feature_names = [c for c in spark.table('user_features').drop('user_id').columns],
    rename_outputs = {c:f'user__{c}' for c in spark.table('user_features').columns} # rename fields to avoid name collisions in when assembling full feature set
    ),
  # product features
  feature_store.FeatureLookup(
    table_name = f"{config['database']}.product_features",
    lookup_key = ['product_id'],
    feature_names = [c for c in spark.table('product_features').drop('product_id').columns],
    rename_outputs = {c:f'product__{c}' for c in spark.table('product_features').columns} # rename fields to avoid name collisions in when assembling full feature set
    )
  ]

# COMMAND ----------

# MAGIC %md We can now use these instructions to retrieve features from the feature store for each labeled instance.  Please note that we are leaving the *user_id* and *product_id* fields in the dataset to help with a later stage of model evaluation but these values will be excluded from the model-training pipeline so that we aren't memorizing user and product combinations:

# COMMAND ----------

# DBTITLE 1,Connect to Feature Store
 fs = FeatureStoreClient()

# COMMAND ----------

# DBTITLE 1,Retrieve Features for Labeled Instances
# define training set
training_set = fs.create_training_set(
      labels,
      feature_lookups=feature_lookups,
      label='label',
      exclude_columns=['user_id','product_id','rating']
      )

# get features and label values
features_and_labels = training_set.load_df()

# display labeled feature set
display(features_and_labels)

# COMMAND ----------

# MAGIC %md Before proceeding, we can divide our data into training and testing datasets to support model training and evaluation. You might be curious why we are doing this when we earlier discussed how the data were pre-split through designations in the *eval_set* field found in the orders table. Here's the thinking on this....
# MAGIC
# MAGIC Many recommendation engines are evaluated on their ability to predict the items a user will select within a ranked sequence.  In that scenario, you want to put all the items in the predicted purchase in one set or the other.  But here, we are training our model to predict whether or not a value will be above or below a threshold.  It's a slightly different question and one that isn't dependent on us holding out all the items in a purchase.
# MAGIC
# MAGIC That said, when we eventually evaluate our model using MAP@K, this metric does require us to collect all the items in a given purchase event.  In that part of this notebook, you will see us tap into the pre-defined split. But at this stage, we'll just use a standard train/test split approach:
# MAGIC
# MAGIC **NOTE** In order to train a model for fast, real-time inference, we currently prefer to work with the non-distributed version of XGBoost. This requires us to submit our data as part of a pandas dataframe which is not as scalable as the Spark dataframe supported by Spark-distributed XGBoost. If you find you have trouble getting your dataset into memory as a pandas dataframe, consider increasing the size of the nodes in your cluster.  If you have maxed out that option, then consider taking a randomly selected subset of your data until the data volume is reduced sufficiently to get it into a single instances' memory.

# COMMAND ----------

# DBTITLE 1,Split Labeled Instances into Train & Test Sets 
# set holdout for test and validate sets
holdout = 0.30

# convert dataframe to pandas
features_and_labels_pd = features_and_labels.toPandas()

# split the data
(train_pd, test_validate_pd) = train_test_split(features_and_labels_pd, test_size=holdout)
(test_pd, validate_pd) = train_test_split(test_validate_pd, test_size=0.5)

# display row counts
print(f"Total Records:    {features_and_labels_pd.shape[0]}")
print(f"Train Records:    {train_pd.shape[0]}")
print(f"Test Records:     {test_pd.shape[0]}")
print(f"Validate Records: {validate_pd.shape[0]}")

# COMMAND ----------

# MAGIC %md ##Step 3: Define Model
# MAGIC
# MAGIC For our model, we'll make use of [XGBoostClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier), a very popular library for classification tasks.  As this library doesn't fully support categorical variables, we will define one-hot encoding transformations for our categorical fields and then wrap this transformation logic within a column transformation:

# COMMAND ----------

# DBTITLE 1,Identify Categorical & Numerical Features
# identify categorical and numerical features
label_col = 'label'
categorical_cols = ['product__department','product__aisle']

numerical_cols = [c for c in features_and_labels.columns if c not in [label_col] + categorical_cols]

# display feature categories
print(f"Categorical Fields:  {categorical_cols}")
print('\n')
print(f"Numerical Fields:  {numerical_cols}")

# COMMAND ----------

# DBTITLE 1,Encode Categorical Features
# define categorical encoding logic
ohe = OneHotEncoder(drop='first', handle_unknown='ignore')

# assemble column transformation logic
col_trans = ColumnTransformer(
  [ ('onehot', ohe, categorical_cols), ('passthru', 'passthrough', numerical_cols) ],
  remainder='drop'
  )

# COMMAND ----------

# MAGIC %md We can then define a pipeline to combine our column transformations with our model:

# COMMAND ----------

# DBTITLE 1,Define Model Pipeline
# assemble pipeline to combine transformations with model
pipe = Pipeline([
  ('transformations', col_trans),
  ('model', XGBClassifier())
  ])

# COMMAND ----------

# MAGIC %md Notice that our model has been defined with all default hyperparameter values.  In the next step of this notebook, you will see us update this stage of the pipeline with a parameterized model as part of our tuning and final training actions.

# COMMAND ----------

# MAGIC %md ##Step 4: Tune the Model
# MAGIC
# MAGIC The XGBoostClassifier is a very powerful model type but it is highly configurable.  Typically, we don't know what the right values for many of the parameters might be, though we might anticipate a range of values for different hyperparameter values that should yield good results. 
# MAGIC
# MAGIC To discover an ideal combination of values within such a range, we might define a search space and then turn over the task of discovering an ideal combination of hyperparameter values to a utility such as [hyperopt](https://docs.databricks.com/machine-learning/automl-hyperparam-tuning/index.html) that can intelligently iterate through the space for us:

# COMMAND ----------

# DBTITLE 1,Define Search Space
search_space = {
    'n_estimators' : hp.quniform('n_estimators', 1, 5, 1)           # num of trees
    ,'max_depth' : hp.quniform('max_depth', 1, 5, 1)                 # depth of trees 
    ,'learning_rate' : hp.uniform('learning_rate', 0.01, 0.40)      # learning rate for XGBoost
    ,'min_child_weight' : hp.quniform('min_child_weight', 1, 20, 1) # minimum number of instances per node
    }
    
  # more info on search spaces: http://hyperopt.github.io/hyperopt/getting-started/search_spaces/

# COMMAND ----------

# MAGIC %md We can then write a function to receive a set of parameters selected from this search space and train and evaluate a model using these values. This function will return a metric that hyperopt will seek to minimize.  Because we are using the [average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn-metrics-average-precision-score) metric, which estimates the area under a precision-recall curve and improves as it increases, we will flip this into a negative value as we return it to hyperopt:

# COMMAND ----------

# DBTITLE 1,Define Function to Evaluate Model
def evaluate_model(hyperparams):

  # accesss replicated input data
  X_train = X_train_broadcast.value
  y_train = y_train_broadcast.value
  X_validate = X_validate_broadcast.value
  y_validate = y_validate_broadcast.value  

  # configure model parameters (convert supplied floats to ints if needed)
  params = hyperparams
  if 'n_estimators' in params: params['n_estimators']=int(params['n_estimators'])
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   
  if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight']) 

  # assign hyperparameters to model
  pipe.steps[-1] = ('model', XGBClassifier(**params))

  # fit model
  pipe.fit(X_train, y_train)
  
  # predict
  y_prob = pipe.predict_proba(X_validate)
  
  # score
  model_ap = average_precision_score(y_validate, y_prob[:,1])
  mlflow.log_metric('avg precision', model_ap)  # record actual metric with mlflow run
  
  # invert metric for hyperopt
  loss = -1 * model_ap  
  
  # return results
  return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md Notice that our function expects several broadcast variables.  Broadcast variables allow us to replicate copies of data to the workers in our cluster.  We will run hyperopt in a distributed manner to reduce the time it takes us to explore the search space.  These copies will help us minimize the overhead of transferring our train and validate datasets with each iteration:

# COMMAND ----------

# DBTITLE 1,Broadcast Data to Workers
X_train_broadcast = sc.broadcast(train_pd.drop('label', axis=1))
y_train_broadcast = sc.broadcast(train_pd['label'])

X_validate_broadcast = sc.broadcast(validate_pd.drop('label', axis=1))
y_validate_broadcast = sc.broadcast(validate_pd['label'])  

# COMMAND ----------

# MAGIC %md With our evaluation function and supporting variables in place, we can now tune our model using hyperopt.  Please note that we have identified a relatively small number of iterations here.  You might get better results with a higher number of iterations.  Please also note that we have set the number of parallel evaluations to a relatively low number that ensures we get multiple cycles of evaluations/learning and that we don't exceed the number of executors in our cluster.  For more info about hyperopt best practices, please review this [blog](https://www.databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html):

# COMMAND ----------

# DBTITLE 1,Perform Model Tuning
# perform evaluation
with mlflow.start_run(run_name='tuning'):
  argmin = fmin(
    fn=evaluate_model,
    space=search_space,
    algo=tpe.suggest,  # algorithm controlling how hyperopt navigates the search space
    max_evals=12, # max number of iterations
    trials=SparkTrials(parallelism=4), # how many evals to perform at a time across the cluster
    verbose=True
    )
  
  # print optimal hyperparameter values
  print('\n')
  print(space_eval(search_space, argmin))

# COMMAND ----------

# MAGIC %md ##Step 5: Finalize the Model
# MAGIC
# MAGIC Having optimized our hyperparameter values, let's train a final version of the model with those and evaluate against the test dataset that was withheld from model tuning:

# COMMAND ----------

# DBTITLE 1,Train & Evaluate Model Using Optimized Hyperparameter Values
with mlflow.start_run(run_name='training'):

  # assemble training inputs
  X = pd.concat([
      train_pd.drop('label', axis=1), 
      validate_pd.drop('label', axis=1)
      ])
  y = pd.concat([
      train_pd['label'],
      validate_pd['label']
    ])

  # clean up parameters (and log them)
  params = space_eval(search_space, argmin)
  if 'n_estimators' in params: params['n_estimators']=int(params['n_estimators'])
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   
  if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight']) 
  mlflow.log_params(params) # log parameters

  # fit model
  pipe.steps[-1] = ('model', XGBClassifier( **params ))
  pipe.fit(X, y)

  # evaluate model
  y_prob = pipe.predict_proba(test_pd.drop('label', axis=1))
  model_ap = average_precision_score(test_pd['label'], y_prob[:,1])
  mlflow.log_metric('avg precision', model_ap) # log evaluation metric

  print(model_ap)

# COMMAND ----------

# MAGIC %md If we've avoided overfitting, our model should produce an evaluation metric value that's similar to the one displayed in the hyperopt output (though hyperopt will display a value that's been multiplied by -1 per the earlier discussion).
# MAGIC
# MAGIC Assuming the metric is okay, we will want to re-persist our model using alternative logic for the *predict* method.  This is because when we deploy our model behind things like the Databricks Model Serving interface, the default method that will be called is the *predict* method on our pipeline.  This method will return a 0 or 1 based on an underlying probability.  But what we want is that probability, not the class that it indicates.  To do this, we need to wrap our model with a class definition that overwrites the *predict* method.  
# MAGIC
# MAGIC Notice too that when we log this model, we are doing so through the feature store. We didn't bother with this above because the previous models being trained were only being used for tuning & evaluation.  But with this one, we want to log the instructions on how features should be retrieved to assist us with our later deployment:
# MAGIC

# COMMAND ----------

# DBTITLE 1,Define Class Wrapper
class MySklearnModel(mlflow.pyfunc.PythonModel):

  # initialize model based on another model
  def __init__(self, model):
    self.model = model

  # overwrite predict to return probability of positive class
  def predict(self, context, inputs):
    return self.model.predict_proba(inputs)[:,1]

wrapped_model = MySklearnModel(pipe)

# COMMAND ----------

# DBTITLE 1,Log Wrapped Model
with mlflow.start_run(run_name='deployment ready'):

  # log model to mlflow with featurestore metadata
  fs.log_model(
    wrapped_model,
    'model',
    flavor=mlflow.pyfunc,
    registered_model_name=config['model_name'],
    training_set=training_set # contains feature store metadata
    )


# COMMAND ----------

# MAGIC %md Finally, we will flag this model as production ready.  In a real-world scenario, you'd want to do this through a structured, possibly human-driven process involving review and integration testing.  But for this demo, it's best to just programmatically flip the model into production-ready status:

# COMMAND ----------

# DBTITLE 1,Elevate Model to Production Status
# connect to mlflow
client = mlflow.tracking.MlflowClient()

# identify model version in registry
latest_model_info = client.search_model_versions(f"name='{config['model_name']}'")[0]
model_version = latest_model_info.version
model_status = latest_model_info.current_stage

# move model to production status and archive existing production versions
if model_status.lower() != 'production':
  client.transition_model_version_stage(
    name=config['model_name'],
    version=model_version,
    stage='production',
    archive_existing_versions=True
    ) 

# COMMAND ----------

# MAGIC %md ##Optional: Explore MAP@K
# MAGIC
# MAGIC In an [earlier recommender](https://www.databricks.com/blog/2023/01/06/products-we-think-you-might-generating-personalized-recommendations.html) developed against this same dataset, we used the metric mean average precision @ k (also known as MAP@k) with a k value of 10, to evaluate the recommender. This metric ignores the precise scores associated with a recommended product and instead focuses on the position of a product in a sorted set of recommendations.  The idea is that if we are presenting customers with product recommendations, the ones they want should be closer to the top of the ranked list.  If we achieve that, we will receive a higher MAP@k score (where k indicates the size of the list).
# MAGIC
# MAGIC To evaluate our classification model using this same metric, we need to first generate k-number of product recommendations for our customers.  While that sounds straightforward, the reality is that for 200K users and nearly 50K products, we'd need to evaluate about 10-trillion user-product combinations which would be either incredibly slow or very expensive to perform.  So, we'll need to find a short cut.
# MAGIC
# MAGIC The simplest shortcut is to grab a random sample of users and evaluate MAP@K for that subset.  Here we'll use a 1% sample to reduce the number of customers down to 2,000, leaving us with about 100-million combinations to consider.  It's still a lot but it should provide a good balance between an expensive exhaustive evaluation and a reasonably accurate evaluation metric:
# MAGIC

# COMMAND ----------

# DBTITLE 1,Assemble Evaluation Set of All Possible Products for User Subset
user_sample_ratio = 0.01

# get features for users in test dataset
users = spark.table('user_features').select('user_id').sample(user_sample_ratio)

# get features for all products
products = spark.table('product_features').select('product_id')

# assemble user-product dataset for evaluation
eval_set = users.crossJoin(products)

# COMMAND ----------

# MAGIC %md Now we can use our deployed model to identify the top 10 recommended products for our selected users.  Notice that we are using the Databricks Feature Store's *score_batch* method to retrieve features for the provided seat of feature lookup keys and then generate scores for these given an identified model.  
# MAGIC
# MAGIC The result of this step is a standard Spark dataframe with a field called prediction that represents our predicted score.  We can manipulate this dataframe like we would any other to produce our set of top-10 predicted products for each user:

# COMMAND ----------

# DBTITLE 1,Identify Top 10 Predicted Products
# determine number of items to recommend for each user
k = 10

# get predictions for users
predictions = (
  fs
    .score_batch(f"models:/{config['model_name']}/production", eval_set) # retrieve features for user-products & score
    .select('user_id','product_id','prediction') # get ids and prediction (probability)
    .withColumn('product_rank', fn.expr('row_number() over(partition by user_id order by prediction desc)')) # rank products based on probabilities
    .filter(f'product_rank <= {k}') # limit to k products
    .withColumn('products', fn.expr('collect_list(product_id) over(partition by user_id order by product_rank)')) # assemble ordered list of products
    .filter(f'size(products) = {k}')
    .withColumn('prediction', fn.col('products').cast('array<double>'))
    .select('user_id','prediction')
  ).cache()

display(predictions)

# COMMAND ----------

# MAGIC %md We can then look at the product selections in the last set of purchases as identified by labels assigned to the *eval_set* field.  While separate *train* and *test* sets are identified in that field, we can combine these for this evaluation:

# COMMAND ----------

# DBTITLE 1,Get Actual Product Selections
actuals = (
  spark
    .table('orders')
    .filter("eval_set != 'prior'") # get holdout data
    .join(
      spark.table('order_products'),
      on='order_id',
      how='inner'
      )
    .withColumn('products', fn.expr("collect_list(product_id) over(partition by user_id order by add_to_cart_order asc)"))
    .filter(fn.expr("size(products)<=10"))
    .groupBy('user_id')
      .agg(
        fn.max('products').alias('products')
        )
    .withColumn('label', fn.col('products').cast('array<double>'))
    .select('user_id','label')
    )

display(actuals)

# COMMAND ----------

# MAGIC %md And using these data calculate our metric.  Notice that we are taking advantage of the pyspark [RankingEvaluator](https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.ml.evaluation.RankingEvaluator.html) to perform this calculation.  This is because there is not a widely accessible MAP@K metric in the sklearn library.  This decision to use a Spark MLLib evaluator motivated us to assemble our recommendations and actuals as arrays in prior cells: 

# COMMAND ----------

# DBTITLE 1,Calculate MAP@10
# evaluate the predictions
eval = RankingEvaluator( 
  predictionCol='prediction',
  labelCol='label',
  metricName='precisionAtK',
  k=k
  )

eval.evaluate( predictions.join(actuals, on='user_id') )

# COMMAND ----------

# MAGIC %md From these results, it doesn't appear we are achieving as high of a MAP@K score as the ALS model which achieved a 0.05. It is possible that for this dataset, the matrix factorization model produces better recommendations.  Alternatively, it may be that our simple and limited approach to feature generation does not provide the model sufficient information to make as good of predictions.  
# MAGIC
# MAGIC Either way, it's important to keep in mind that an offline evaluation of a recommender is always questionable.  The best evaluation of a recommender that shows promising results is to test it with an online experiment where you can assess the model based on actual user responses.

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
