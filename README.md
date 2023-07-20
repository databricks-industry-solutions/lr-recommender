![image](https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo_wide.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

## Introduction

Recommender systems are becoming increasing important as companies seek better ways to select products to present to end users. In this solution accelerator, we will explore a form of collaborative filter based on binary classification.  In this recommender, we will leverage user and product features to predict product preferences. Predicted preferences will then be used to determine the sequence within which a given set of products are presented to a given user, forming the basis of the recommendation.

The recommender model will be deployed for real-time inference using Databricks Model Serving.  To support the rapid retrieval of features based on a supplied set of users and products, a Databricks Online Feature Store will be employed.  Before proceeding with the notebooks in this accelerator, it is recommended you deploy an instance of either AWS DynamoDB or Azure CosmosDB depending on which cloud you are using. (The AWS details are provided [here](https://www.cedarwoodfurniture.com/garden-bench.html) while the Azure details are provided [here](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/online-feature-stores).) An additional consideration with Model Serving is that it is not currently available on GCP or within every region supporting Databricks on AWS and Azure.  Before moving forward with these notebooks, please verify you are in a region that supports this feature.  (Details on supported regions for [AWS](https://docs.databricks.com/resources/supported-regions.html#supported-regions-list) and [Azure](https://learn.microsoft.com/en-us/azure/databricks/resources/supported-regions#--supported-regions-list) are found here.)

___
<bryan.smith@databricks.com>

___

&copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| azure-cosmos-spark                                 | Apache Spark Connector for Azure Cosmos DB      | MIT        | https://github.com/Azure/azure-cosmosdb-spark                     |

## Getting started

Although specific solutions can be downloaded as .dbc archives from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries. 

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

To start using a solution accelerator in Databricks simply follow these steps: 

1. Clone solution accelerator repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
2. Attach the `RUNME` notebook to any cluster and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. The job configuration is written in the RUNME notebook in json format. 
3. Execute the multi-step-job to see how the pipeline runs. 
4. You might want to modify the samples in the solution accelerator to your need, collaborate with other users and run the code samples against your own data. To do so start by changing the Git remote of your repository  to your organization’s repository vs using our samples repository (learn more). You can now commit and push code, collaborate with other user’s via Git and follow your organization’s processes for code development.

The cost associated with running the accelerator is the user's responsibility.


## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
