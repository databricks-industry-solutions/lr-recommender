# Databricks notebook source
# MAGIC %md The purpose of this notebook is to introduce the classification recommender solution accelerator and to provide access to configuration information for the notebooks supporting it. You may find this notebook at https://github.com/databricks-industry-solutions/lr-recommender.

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC
# MAGIC Recommender systems are becoming increasing important as companies seek better ways to select products to present to end users. In this solution accelerator, we will explore a form of collaborative filter based on binary classification.  In this recommender, we will leverage user and product features to predict product preferences. Predicted preferences will then be used to determine the sequence within which a given set of products are presented to a given user, forming the basis of the recommendation.
# MAGIC
# MAGIC The recommender model will be deployed for real-time inference using Databricks Model Serving.  To support the rapid retrieval of features based on a supplied set of users and products, a Databricks Online Feature Store will be employed.  Before proceeding with the notebooks in this accelerator, it is recommended you deploy an instance of either AWS DynamoDB or Azure CosmosDB depending on which cloud you are using. (The AWS details are provided [here](https://www.cedarwoodfurniture.com/garden-bench.html) while the Azure details are provided [here](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/online-feature-stores).)
# MAGIC
# MAGIC **NOTE** For our development, we are making use of Azure and therefore have installed the [Azure Cosmos DB Spark 3 OLTP Connector for SQL API](https://github.com/Azure/azure-sdk-for-java/blob/main/sdk/cosmos/azure-cosmos-spark_3-2_2-12/README.md#download) in order to connect from Databricks to CosmosDB.  As a Java JAR, the connector must be installed as either a [cluster or workspace library](https://learn.microsoft.com/en-us/azure/databricks/libraries/). We used *azure-cosmos-spark_3-3_2-12 version 4.17.0*. at the time of notebook development. If you use the cluster or workflow created by the `./RUNME` notebook, the package is already configured for you.
# MAGIC
# MAGIC An additional consideration with Model Serving is that it is not currently available on GCP or within every region supporting Databricks on AWS and Azure.  Before moving forward with these notebooks, please verify you are in a region that supports this feature.  (Details on supported regions for [AWS](https://docs.databricks.com/resources/supported-regions.html#supported-regions-list) and [Azure](https://learn.microsoft.com/en-us/azure/databricks/resources/supported-regions#--supported-regions-list) are found here.)

# COMMAND ----------

# MAGIC %md ## Configuration Settings
# MAGIC
# MAGIC The following configuration settings are used throughout the remaining notebooks in this accelerator:

# COMMAND ----------

# DBTITLE 1,Instantiate Config Variable 
if 'config' not in locals().keys():
  config = {}

# COMMAND ----------

# DBTITLE 1,Database
config['database'] = 'lrrec'

# create database if not exists
_ = spark.sql('create database if not exists {0}'.format(config['database']))

# set current datebase context
_ = spark.catalog.setCurrentDatabase(config['database'])

# COMMAND ----------

# DBTITLE 1,Storage
config['mount_point'] ='/tmp/instacart_lr_rec'

config['products_path'] = config['mount_point'] + '/bronze/products'
config['orders_path'] = config['mount_point'] + '/bronze/orders'
config['order_products_path'] = config['mount_point'] + '/bronze/order_products'
config['aisles_path'] = config['mount_point'] + '/bronze/aisles'
config['departments_path'] = config['mount_point'] + '/bronze/departments'

# COMMAND ----------

# DBTITLE 1,Model
config['model_name'] = 'classification_recommender'

# COMMAND ----------

# DBTITLE 1,Feature Store
config['cosmosdb_uri'] = 'https://brysmi-lrrrec.documents.azure.com:443/'

config['scope_readonly'] = 'recommender-readonly'
config['scope_readwrite'] = 'recommender-readwrite'
config['secret_prefix'] = 'onlinefs'

# COMMAND ----------

# DBTITLE 1,Databricks url and token
import os
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
config['databricks token'] = ctx.apiToken().getOrElse(None)
config['databricks url'] = ctx.apiUrl().getOrElse(None)
os.environ['DATABRICKS_TOKEN'] = config["databricks token"]
os.environ['DATABRICKS_URL'] = config["databricks url"]

# COMMAND ----------

# DBTITLE 1,mlflow experiment
import mlflow
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/lr_rec'.format(username))

# COMMAND ----------

config['serving_endpoint_name'] = 'classification_recommender'

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
