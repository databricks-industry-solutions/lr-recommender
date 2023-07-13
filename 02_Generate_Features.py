# Databricks notebook source
# MAGIC %md The purpose of this notebook is to prepare the features for use in the classifier recommender. You may find this notebook at https://github.com/databricks-industry-solutions/lr-recommender.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC In this notebook, we'll calculate various user- and product-level features that we will use to predict user preferences.  We have broad flexibility in terms of the kinds of features we generate within each of these categories, but its important to remember that we are trying to predict preferences for a broader range of products than just those that have been purchased by an individual customer.  With that in mind, we want to avoid deriving features for specific user-product combinations.
# MAGIC
# MAGIC **NOTE** We will keep our feature generation efforts here pretty lightweight.  A more exhaustive approach to user and product feature generation might yield better predictive results.

# COMMAND ----------

# DBTITLE 1,Get Config Info
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn

from databricks.feature_store import FeatureStoreClient
from databricks.feature_store.online_store_spec import AzureCosmosDBSpec

# COMMAND ----------

# MAGIC %md ##Step 1: Define Features
# MAGIC
# MAGIC Our first step will be to calculate features.  Quite often, we will limit our feature generation to activity that has occurred within a specific period in time, *e.g.* last 30-days, last year, *etc.*, or within a specific position within a sequence, *i.e.* the last order. This dataset lacks date information and but the dataset originators have marked each order with an *eval_set* value.  Orders earlier within a customer's sequence of orders are marked as *prior* while the last order in a sequence is marked with either *train* or *test*.  (We'll ignore the train/test split for now and perform our own split later.)
# MAGIC
# MAGIC For our product features, our metrics will examine how frequently it is that products are ordered and re-ordered relative to all products in the system.  We will also include a few categorical features such as the aisle and department to which a product is assigned:
# MAGIC
# MAGIC **NOTE** We are using the department and aisle name designations and not the integer IDs associated with these because the sklearn pipeline we'll use later has some funny behaviors when we employ integers, even when they are presented as strings.

# COMMAND ----------

# DBTITLE 1,Define Product Features
product_features = (
  spark
    .table('orders')
    .filter("eval_set='prior'")
    .select('order_id')
    .join(
      spark.table('order_products'),
      on='order_id',
      how='inner'
      )
    .groupBy('product_id')
      .agg(
        fn.count('*').alias('orders'),
        fn.sum('reordered').alias('reorders'),
        fn.avg('add_to_cart_order').alias('avg_add_to_cart_order')
        )
      .withColumn('orders_ratio', fn.expr('orders / SUM(orders) OVER()'))
      .withColumn('reorders_ratio', fn.expr('reorders / SUM(reorders) OVER()'))
      .alias('a')
    .join(
      spark.table('products').alias('b'),
      on='product_id'
      )
    .join(
      spark.table('departments'),
      on='department_id',
      how='left'
      )
    .join(
      spark.table('aisles'),
      on='aisle_id',
      how='left'
      )
    .selectExpr(
      'a.product_id','orders_ratio','reorders_ratio','avg_add_to_cart_order',
      'aisle','department' # don't use the integer category assignments as these can cause problems during encoding
      )
  )

display(product_features)

# COMMAND ----------

# MAGIC %md For users, we'll examine order frequencies and cart sizes:

# COMMAND ----------

# DBTITLE 1,Define User Features
user_features = (
  spark
    .table('orders')
    .filter("eval_set='prior'")
    .select('order_id','user_id','days_since_prior_order')
    .join(
      spark.table('order_products'),
      on='order_id'
    )
    .groupBy('user_id','order_id')
      .agg(
        fn.count('*').alias('ordered_items'),
        fn.sum('reordered').alias('reordered_items'),
        fn.max('reordered').alias('includes_reorders'),
        fn.first('days_since_prior_order').alias('days_since_prior_order')
        )
    .groupBy('user_id')
      .agg(
        fn.count('*').alias('orders'),
        fn.expr('count_if(includes_reorders=1)').alias('orders_with_reorders'),
        fn.sum('ordered_items').alias('ordered_items'),
        fn.sum('reordered_items').alias('reordered_items'),
        fn.avg('ordered_items').alias('avg_basket_ordered_items'),
        fn.avg('reordered_items').alias('avg_basket_reordered_items'),
        fn.avg('days_since_prior_order').alias('avg_days_since_prior_order')
        )
      .withColumn('orders_ratio',fn.expr('orders / SUM(orders) OVER()'))
      .withColumn('orders_with_reorders_ratio',fn.expr('orders_with_reorders / SUM(orders_with_reorders) OVER()'))
      .withColumn('ordered_items_ratio',fn.expr('ordered_items / SUM(ordered_items) OVER()'))
      .withColumn('reordered_items_ratio',fn.expr('reordered_items / SUM(reordered_items) OVER()'))
      .select(
        'user_id','orders_ratio','orders_with_reorders_ratio','ordered_items_ratio','reordered_items_ratio',
        'avg_basket_ordered_items','avg_basket_reordered_items','avg_days_since_prior_order'
        )
  )

display(user_features)

# COMMAND ----------

# MAGIC %md ##Step 2: Persist to Feature Store
# MAGIC
# MAGIC To help us with use of our features during both model training and model inference, we will persist these features to our [Feature Store](https://docs.databricks.com/machine-learning/feature-store/index.html).  The feature store provides us the means to easily capture data about our features and perform feature lookup.  To get started, we connect to the Databricks Feature Store as follows:

# COMMAND ----------

# DBTITLE 1,Connect to Feature Store
# instantiate feature store client
fs = FeatureStoreClient()

# COMMAND ----------

# MAGIC %md We then persist our features to the feature store as tables as follows:

# COMMAND ----------

# DBTITLE 1,Persist User Features
# create feature store table (we will receive a warning with each call after the table has been created)
try: # if feature store does not exist
  fs.get_table(f"{config['database']}.user_features")
except: # create it now
  pass
  _ = (
    fs
      .create_table(
        name=f"{config['database']}.user_features", # name of feature store table
        primary_keys= 'user_id', # name of keys that will be used to locate records
        schema=user_features.schema, # schema of feature set as derived from our feature_set dataframe
        description='user features for product recommendation' 
      )
    )

# persist feature set data in feature store
_ = (
  fs
    .write_table(
      name=f"{config['database']}.user_features",
      df = user_features,
      mode = 'overwrite' 
    )
  )

# display features
display(
  fs.read_table(f"{config['database']}.user_features")
  )

# COMMAND ----------

# DBTITLE 1,Persist Product Features
# create feature store table (we will receive a warning with each call after the table has been created)
try: # if feature store does not exist
  fs.get_table(f"{config['database']}.product_features")
except: # create it now
  pass
  _ = (
    fs
      .create_table(
        name=f"{config['database']}.product_features", # name of feature store table
        primary_keys= 'product_id', # name of keys that will be used to locate records
        schema=product_features.schema, # schema of feature set as derived from our feature_set dataframe
        description='product features for product recommendation' 
      )
    )

# persist feature set data to feature store
_ = (
  fs
    .write_table(
      name=f"{config['database']}.product_features",
      df = product_features,
      mode = 'overwrite' 
    )
  )

# display features
display(
  fs.read_table(f"{config['database']}.product_features")
  )

# COMMAND ----------

# MAGIC %md ##Step 3: Sync with Online Feature Store
# MAGIC
# MAGIC We can now publish our data to an online feature store. The online feature store is either a high-performance relational database or document store that's accessible to our model serving layer.  Because we wish to leverage the Databricks serverless real-time inference capability (recently renamed as Databricks *Model Serving*) for this, we are locked into the use of a [CosmosDB document store](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/online-feature-stores) in the Azure cloud and a [DynamoDB document store](https://docs.databricks.com/machine-learning/feature-store/online-feature-stores.html) in AWS. (The online feature store is not yet available in the Google cloud as of the time of notebook development.)
# MAGIC
# MAGIC Because we are demonstrating this solution accelerator in the Azure cloud, we will be setting up an Azure CosmosDB document store.  The steps for deploying an Azure CosmosDB document store are found [here](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/quickstart-portal).  The key items to consider are:
# MAGIC </p>
# MAGIC
# MAGIC * the document store should be deployed to the same region as your Databricks cluster
# MAGIC * the Core (SQL) API (aka *Azure Cosmos DB for NoSQL*) should be specified during CosmosDB deployment
# MAGIC * network connectivity should be set to *All Networks* on the CosmosDB service so that the Databricks service can communicate directly to it
# MAGIC
# MAGIC Once your CosmosDB document store has been deployed, be sure to get one [authorization key](https://learn.microsoft.com/en-us/azure/cosmos-db/secure-access-to-data?tabs=using-primary-key#primary-keys) from the CosmosDB service with read-only access to the store and another with read-write access to the store. You'll need these in later steps.  You will also need to capture the CosmosDB URI and record it in notebook *00*.

# COMMAND ----------

# DBTITLE 1,Set Online Feature Store Information
print(f"cosmosdb_uri:\t{config['cosmosdb_uri']}")

# COMMAND ----------

# MAGIC %md Before proceeding, it's a good idea to make sure you've configured your Databricks cluster to use the latest [Azure Cosmos DB Spark 3 OLTP Connector for SQL API](https://github.com/Azure/azure-sdk-for-java/blob/main/sdk/cosmos/azure-cosmos-spark_3-2_2-12/README.md#download).  As a Java JAR, it must be installed as either a [cluster or workspace library](https://learn.microsoft.com/en-us/azure/databricks/libraries/). 
# MAGIC
# MAGIC **NOTE** We used *azure-cosmos-spark_3-3_2-12 version 4.17.0*. at the time of notebook development. If you use the cluster or workflow created by the `./RUNME` notebook, the package is already configured for you.

# COMMAND ----------

# MAGIC %md With the Azure CosmosDB document store deployed and the library installed, we now need to record the read-only and read-write authentication keys for the store as [Databricks secrets](https://learn.microsoft.com/en-us/azure/databricks/security/secrets/secrets#create-a-secret-in-a-databricks-backed-scope). In an Azure environment, you can create either a Databricks-backed scope or an Azure Key Vault scope.  In this demo, we have employed a Databricks-backed scope to keep things simpler.
# MAGIC
# MAGIC ___
# MAGIC
# MAGIC To setup a secret, you need to make use of the Databricks CLI. To use the CLI, you first need to install and configure it to your local system, and to do that, you'll need to follow the instructions provided [here](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/). (While the CLI runs on your local system, it creates the secrets in the environment for which it has been configured.  It is critical that you configure your installation of the Databricks CLI to point to the environment where you are running these notebooks.)
# MAGIC
# MAGIC After it's been configured, you'll need to setup two secret scopes, one to hold your read-only key and the other to hold your read-write key.  For example, you might create scopes as follows:
# MAGIC
# MAGIC ```
# MAGIC databricks secrets create-scope --scope recommender-readonly
# MAGIC databricks secrets create-scope --scope recommender-readwrite
# MAGIC ```
# MAGIC
# MAGIC Once the scopes are defined, you now need to place the approrpriate authentication keys in them.  Each key will use a set prefix that you will define.  Here, we are using a prefix of `onlinefs`. Please note, the remainder of the key name should be recorded as *authorization-key*:
# MAGIC
# MAGIC ```
# MAGIC databricks secrets put --scope recommender-readonly --key onlinefs-authorization-key
# MAGIC databricks secrets put --scope recommender-readwrite --key onlinefs-authorization-key
# MAGIC ```
# MAGIC As you enter each command, you will be prompted to select a text editor.  Choose the one you are most familiar with and follow the instructions, pasting the appropriate CosmosDB authentication key in each. Be sure to record your scope names and prefix in notebook *00*.
# MAGIC
# MAGIC ___
# MAGIC
# MAGIC If you don't have Databricks CLI configured, you can use the automation script in `./RUNME` to help expedite secret scope setup. See the `./RUNME` notebook for details.

# COMMAND ----------

# DBTITLE 1,Set Secrets Information
print(f"scope_readonly:  {config['scope_readonly']}")
print(f"scope_readwrite: {config['scope_readwrite']}")
print(f"secret_prefix:   {config['secret_prefix']}")

# COMMAND ----------

# MAGIC %md With the service behind our online feature store deployed and configuration settings used to connect us to this service captured in notebook *00*, we can now define our online feature store specification:

# COMMAND ----------

# DBTITLE 1,Define Online Feature Store Spec
online_store_spec = AzureCosmosDBSpec(
  account_uri=config['cosmosdb_uri'],
  read_secret_prefix=f"{config['scope_readonly']}/{config['secret_prefix']}",
  write_secret_prefix=f"{config['scope_readwrite']}/{config['secret_prefix']}"
  )

# COMMAND ----------

# MAGIC %md Using this spec, we can now push updated data to the online feature store as follows:

# COMMAND ----------

# DBTITLE 1,Publish Cart Metrics
_ = fs.publish_table(
  f"{config['database']}.user_features", # offline feature store table where features come from 
  online_store_spec, # specs for connecting to online feature store
  mode = 'merge'
  )

# COMMAND ----------

# DBTITLE 1,Publish Cart-Product Metrics
_ = fs.publish_table(
  f"{config['database']}.product_features", 
  online_store_spec,
  mode = 'merge'
  )

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
