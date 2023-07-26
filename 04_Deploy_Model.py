# Databricks notebook source
# MAGIC %md The purpose of this notebook is to deploy the classifier recommender for real-time inference. You may find this notebook at https://github.com/databricks-industry-solutions/lr-recommender.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC With our model trained, we can now turn our attention to deployment.  While we can use the model to make batch recommendations, we will more likely use a model like this to generate real-time metrics against a provided set of users and products. An application can then use this information to present those products to a user in an appropriate sequence.
# MAGIC
# MAGIC To support real-time interactions with the model, we will deploy it using [Databricks Model Serving](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/).  In order to access this feature, you will need to verify you are in a cloud and a region that supports Model Serving.  (Details on supported regions for [AWS](https://docs.databricks.com/resources/supported-regions.html#supported-regions-list) and [Azure](https://learn.microsoft.com/en-us/azure/databricks/resources/supported-regions#--supported-regions-list) are found here.)

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import os
import pandas as pd 

# COMMAND ----------

# MAGIC %md ##Step 1: Deploy Model
# MAGIC
# MAGIC With our model in production status, we can now deploy it to the Databricks Model Serving infrastructure.  To do this, we need to create a Serving endpoint using [these steps](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints). Be sure to configure the Databricks UI for *Machine Learning* (instead of using the default *Data Science & Engineering* configuration) in order to more easily access the *Serving* icon in the sidebar UI.  When selecting your model, be sure to select the instance using the *\__inference* suffix.  Scale the compute per your requirements (though we used a Medium configuration with *Scale to zero* deselected for our testing).
# MAGIC
# MAGIC Please wait until the endpoint is fully deployed and running before proceeding to the next step:
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/crec_model_serving.PNG'>
# MAGIC

# COMMAND ----------

# MAGIC %md Alternatively, we can use the API instead of the UI to [set up the model serving endpoint](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html#create-model-serving-endpoints). The effect of the following 4 blocks of code is equivalent to the steps described above. We provide this option to showcase automation and to make sure that this notebook can be consistently executed end-to-end without requiring manual intervention.
# MAGIC
# MAGIC To use the Databricks API, you need to create environmental variables named *DATABRICKS_URL* and *DATABRICKS_TOKEN* which must be your workspace url and a valid [personal access token](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/authentication). We have retrieved and set these values up for you in notebook *00* as part of the *config* setting.

# COMMAND ----------

# MAGIC %run ./util/create-update-serving-endpoint

# COMMAND ----------

# DBTITLE 1,Select a model 
model_name = config['model_name']

# identify model version in registry
model_version = mlflow.tracking.MlflowClient().get_latest_versions(name = model_name, stages = ["Production"])[0].version

# COMMAND ----------

# DBTITLE 1,Define model serving config
served_models = [
    {
      "name": config['serving_endpoint_name'],
      "model_name": model_name,
      "model_version": model_version,
      "workload_size": "Medium",
      "scale_to_zero_enabled": True
    }
]
traffic_config = {"routes": [{"served_model_name": model_name, "traffic_percentage": "100"}]}

# COMMAND ----------

# DBTITLE 1,Create or update model serving endpoint
# kick off endpoint creation/update
if not endpoint_exists(config['serving_endpoint_name']):
  create_endpoint(config['serving_endpoint_name'], served_models)
else:
  update_endpoint(config['serving_endpoint_name'], served_models)

# COMMAND ----------

# MAGIC %md ##Step 2: Test Model API
# MAGIC
# MAGIC Before moving away from the model serving endpoint UI, click the Query Endpoint button in the upper right-hand corner of the UI and copy the Python code in the resulting popup.  You can paste a copy of this code in the cell below:

# COMMAND ----------

# DBTITLE 1,Assign Query Endpoint URL Here
MY_QUERYENDPOINT_URL = f"""{config['databricks url']}/serving-endpoints/{config['serving_endpoint_name']}/invocations"""

# COMMAND ----------

# DBTITLE 1,Paste the Query Endpoint | Python Code Here
import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = MY_QUERYENDPOINT_URL
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 
'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')

  return response.json()

# COMMAND ----------

# MAGIC %md Running the pasted code you have now defined a couple functions with which you can test your endpoint:

# COMMAND ----------

# DBTITLE 1,Assemble Set of Keys to Score
# get a user
users = (
  spark
    .table('user_features')
    .select('user_id')
    .limit(1)
  )

# get producs for a given aisle
products = (
  spark
    .table('product_features')
    .select('product_id')
    .filter("aisle='fresh fruits'")
    .limit(10)
  )

# assemble lookup keys to score
training_pd = (
  users
    .crossJoin(products)
  ).toPandas()


display(training_pd)

# COMMAND ----------

# DBTITLE 1,Score the Keys
# retrieve scores from endpoint
scores = score_model(training_pd)

# match scores to keys
training_pd['prediction'] = pd.DataFrame(scores)

# display results
display(
  training_pd
  )

# COMMAND ----------

# MAGIC %md Notice the API returns just the scores for the keys submitted.  To marry these with the user-product ids associated with each, we've added a few lines of code.  You could alter the class wrapper in the prior notebook to do the task internally, though this my complicate the MAP@k evaluation we did earlier.

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
