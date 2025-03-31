# Azure ML documentation

TODO: Move this tutorial to a new repo in NGI, including the Titanic example.


## Table of Contents
- [Azure ML](#azure-ml)
  - [Table of Contents](#table-of-contents)
  - [Tools- and system setup](#tools--and-system-setup)
    - [Azure account](#azure-account)
    - [Azure subscription](#azure-subscription)
    - [Azure CLI](#azure-cli)
    - [VSCode integrated with Azure ML](#vscode-integrated-with-azure-ml)
  - [Azure Assets and Resources](#azure-assets-and-resources)
    - [Workspace](#workspace)
      - [Organising workspaces:](#organising-workspaces)
      - [Setting up a workspace:](#setting-up-a-workspace)
      - [Content stored in a workspace:](#content-stored-in-a-workspace)
      - [Connect to a workspace:](#connect-to-a-workspace)
    - [Compute instance](#compute-instance)
    - [Datastore - including how to reference data in a datastore](#datastore---including-how-to-reference-data-in-a-datastore)
      - [1. Prepare and upload your dataset to Azure blob storage](#1-prepare-and-upload-your-dataset-to-azure-blob-storage)
      - [2. Register the dataset](#2-register-the-dataset)
    - [Environment](#environment)
    - [Models](#models)
    - [Pipelines](#pipelines)
    - [Component](#component)
  - [Training a model in Azure ML](#training-a-model-in-azure-ml)
    - [Typical workflow for training a model in Azure ML](#typical-workflow-for-training-a-model-in-azure-ml)
    - [Complete script for training a model in Azure ML](#complete-script-for-training-a-model-in-azure-ml)
  - [Experiment tracking in Azure ML](#experiment-tracking-in-azure-ml)
    - [MLflow](#mlflow)
    - [Tensorboard](#tensorboard)
  - [Dataset versioning in Azure ML](#dataset-versioning-in-azure-ml)
    - [Register a Dataset in Azure ML](#register-a-dataset-in-azure-ml)
      - [Example: Register a Dataset](#example-register-a-dataset)
    - [View Dataset Versions](#view-dataset-versions)
      - [List All Versions of a Dataset](#list-all-versions-of-a-dataset)
    - [Use a Specific Dataset Version in an Experiment](#use-a-specific-dataset-version-in-an-experiment)
      - [Example: Use a Specific Dataset Version](#example-use-a-specific-dataset-version)
    - [Dataset Versioning in the Experiment Log](#dataset-versioning-in-the-experiment-log)
      - [Retrieve Dataset Information from a Job](#retrieve-dataset-information-from-a-job)
    - [5. Benefits of Dataset Versioning](#5-benefits-of-dataset-versioning)
    - [6. Updating Datasets](#6-updating-datasets)
      - [Register a New Version](#register-a-new-version)
    - [7. Data Asset Management in Azure ML Studio](#7-data-asset-management-in-azure-ml-studio)
    - [8. Integrating with Git and CI/CD](#8-integrating-with-git-and-cicd)
  - [AutoML](#automl)
  - [Example tutorials](#example-tutorials)
    - [Predicting Titanic survival](#predicting-titanic-survival)
    - [Rock joint detection](#rock-joint-detection)
    - [LabOAI](#laboai)
  - [Learning resources](#learning-resources)


---

## Related Azure concepts

We start by defining some relevant Azure concepts. This is not a complete list of all Azure concepts, but it is a good starting point for understanding the Azure ecosystem. For a complete list of Azure concepts, please refer to the [Azure documentation](https://learn.microsoft.com/en-us/azure/?view=azure-cli-latest).

### Azure portal

Azure Portal is the web-based interface for managing Azure resources. It provides a user-friendly way to create, configure, and monitor Azure services, including Azure Machine Learning. You can access the Azure Portal at https://portal.azure.com/.

### Azure ML

Azure ML is a cloud-based service for creating, managing, and deploying machine learning models. It provides a centralized place for data scientists and developers to work with all the artifacts for machine learning, including datasets, training scripts, and trained models. You can create a model in Azure ML or use a model built from an open-source platform, such as PyTorch, TensorFlow, or scikit-learn. Azure ML make it easy to scale your ML training by harnessing powerful cloud compute resources, such as nodes with several GPU's with lots of memory. Azure ML is for individuals and teams implementing `MLOps` within their organization to bring ML models into production in a secure and auditable production environment. You reach Azure ML through http://ml.azure.com.

Azure ML has a comprehensive documentation. In this documentation for a `code academy course` in MLOps and professional ML development in Azure ML we have tried to simplify the most important parts of Azure ML and to extract the parts which currently are most relevant for NGI.

In this tutorial we have listed the parts in a typical MLOps workflow that is facilitated by Azure ML. We will describe and run these operations mainly in the form of Python scripts, but the experiments, the models and other artifacts will be visualised in `Azure ML studio`: http://ml.azure.com.

The main operations carried out in Azure ML Studio are:

- Model processing, training, evaluation by defining reusable ML pipelines. NOTE: You don't need `Airflow` or other orchestration tools to run pipelines in Azure ML. Azure ML has its own orchestration engine.
- Hyperparameter tuning
- Reusable environments for training and deployment
- Model evaluation
- Experiment tracking to mlflow and/or tensorboard
- Dataset versioning
- Model versioning
- Model packaging, registration and deployment
- Monitoring of deployed models
- Retraining of deployed models

MLOps steps not covered by Azure ML can be found in the [MLOps](MLOps.md) document.

There are a few other tools which can be accessed through Azure ML Studio, such as `AutoML`, `Data labeling`, and `Azure ML designer`. These tools are not covered in detail in this document, but they can be useful for specific use cases.

#### Azure ML Designer

Azure ML Designer is a drag-and-drop interface for building machine learning models without writing code. It allows users to create and manage machine learning workflows visually. This can be useful for users who are not familiar with programming or prefer a more visual approach to building models. Azure ML Designer provides a set of pre-built modules for common tasks, such as data preprocessing, model training, and evaluation. Users can connect these modules to create a complete machine learning pipeline. Azure ML Designer is a good tool for quickly prototyping machine learning models and workflows.

Reference: https://learn.microsoft.com/en-us/azure/machine-learning/concept-designer?view=azure-ml-py&tabs=python#overview

#### AutoML

AutoML is a feature in Azure Machine Learning that automates the process of building and tuning machine learning models. It helps users quickly create high-quality models without requiring extensive knowledge of machine learning algorithms or hyperparameter tuning. AutoML can automatically select the best algorithm, preprocess the data, and optimize hyperparameters to achieve the best performance. This can save time and effort for data scientists and developers, allowing them to focus on other aspects of their projects. AutoML is particularly useful for users who are new to machine learning or have limited experience with model development. AutoML can also be used as a quick first step to identify the best algorithms and hyperparameters for a specific problem before moving on to more advanced techniques or custom model development.

Reference: https://learn.microsoft.com/en-us/azure/machine-learning/concept-automated-ml?view=azure-ml-py&tabs=python#overview

#### Data labeling

Azure ML Data Labeling is a feature that helps users annotate and label data for machine learning tasks. It provides a user-friendly interface for creating and managing labeling projects, allowing users to upload datasets, define labeling tasks, and collaborate with labelers. Data Labeling supports various types of data, including images, text, and audio. Users can create custom labeling tasks, such as object detection, image classification, and text classification. The labeled data can then be used to train machine learning models in Azure ML. Data Labeling is particularly useful for users who need to prepare large datasets for supervised learning tasks.

Reference: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-label-data?view=azureml-api-2

#### References:

- https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment?view=azureml-api-2
https://learn.microsoft.com/en-us/azure/machine-learning/?view=azureml-api-2
- https://medium.com/henkel-data-and-analytics/how-to-use-azure-ml-studio-an-eye-opening-model-training-tutorial-for-beginners-from-henkels-data-5035ee10a6d2

### Azure AI Foundry portal

 Azure AI Foundry portal is a unified platform for developing and deploying `generative AI apps` and Azure AI APIs responsibly. It includes a rich set of AI capabilities, simplified user interface and code-first experiences, offering a one-stop shop to build, test, deploy, and manage intelligent solutions. Generative AI development is not the main topic of this document, but it is worth mentioning that Azure AI Foundry portal is a powerful tool for building and deploying generative AI applications. It provides a set of pre-built models and APIs for common tasks, such as text generation, image generation, and speech synthesis. The portal also includes tools for managing and monitoring the performance of your applications. Azure AI Foundry portal is a good choice for users who want to quickly build and deploy generative AI applications without having to worry about the underlying infrastructure.

 Reference: https://learn.microsoft.com/en-us/azure/ai-foundry/?view=azure-ai-foundry-portal&tabs=python-sdk#overview
### Azure DevOps

#### Examples

Throughout this documentation we will often refer to an example NGI project called `RockJointDetection`. This project is a machine learning project that aims to detect rock joints in images. The project is used as an example to demonstrate the functionality of Azure ML and how to use it for machine learning projects. The project is available here in Azure Devops: https://dev.azure.com/ngi001/NGI/_git/syntetic_rock_joint_generation_for_ml

## Tools- and system setup

### Azure account

To use Azure ML you need an Azure account. In NGI all employees have an Azure account. You can check if you have an Azure account by logging into the Azure portal at https://portal.azure.com/. If you are logged in, you have an Azure account. If you are not logged in, you can log in with your NGI email address and password. If you dont have an account you can also create a free account at https://go.microsoft.com/fwlink/?linkid=2227353&clcid=0x409&l=en-us&icid=azurefreeaccount. The free account gives you access to a limited set of Azure services for 12 months. You can use the free account to explore Azure ML and other Azure services.

Reference: https://azure.microsoft.com/en-us/pricing/purchase-options/azure-account?icid=azurefreeaccount

### Azure subscription

An Azure subscription is a logical container used to provision resources in Azure. It holds the details of all your resources, such as compute nodes, virtual machines, storage accounts, and databases. Each Azure subscription is associated with a specific Azure account and can be managed through the Azure portal. You can create multiple subscriptions under a single Azure account to organize and manage your resources based on different projects or departments. For this tutorial we have made a subscription called `ngi_mlops_sandbox`. You can check your subscriptions in the Azure portal by clicking on `Subscriptions` in the top menu. Ask IT services (Christian Demeter) to create an Azure subscription for you if you do not have one.

Once you have an Azure account and subscription, you can create an Azure Machine Learning workspace to start building, training, and deploying machine learning models. More about workspaces can be found in the [Workspace](#workspace) section below.

### Azure CLI

Azure CLI is a command-line tool that provides a set of commands for managing Azure resources. You can use Azure CLI to create and manage Azure resources, such as virtual machines, storage accounts, and Azure Machine Learning workspaces. Azure CLI is available for Windows, macOS, and Linux. You can install Azure CLI on your local machine or use the Azure Cloud Shell, which is a browser-based shell that comes pre-installed with Azure CLI. You can also login into Azure CLI using the standard terminal in VSCode using the command: `az login`. This will open a browser window where you can log in with your Azure account.

To install Azure CLI on your local machine, follow the instructions in the Azure CLI installation guide: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest


Reference: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public

### Azure APP service

Azure App Service is a fully managed platform for building, deploying, and scaling web apps. It provides a range of features, including support for multiple programming languages, built-in authentication and authorization, and integration with Azure DevOps for continuous deployment. Azure App Service is a good choice for hosting web applications, RESTful APIs, and mobile backends. It allows you to focus on your application code while Azure manages the underlying infrastructure.

You can use Azure App Service to host your machine learning models and APIs. This allows you to deploy your models as web services that can be accessed by other applications or users. Azure App Service provides built-in support for scaling, load balancing, and monitoring, making it easy to manage your deployed models.

Reference: https://learn.microsoft.com/en-us/azure/app-service/overview?view=azure-app-service&tabs=python#overview

### Azure ML Python SDKv2

The Azure ML Python SDK is a set of Python libraries that provide a convenient way to interact with Azure Machine Learning services. The SDK allows you to create and manage Azure Machine Learning resources, such as workspaces, compute instances, and datasets, directly from your Python code. The SDK also provides tools for training and deploying machine learning models, as well as tracking experiments and managing model versions. We will mainly create and manage resources in the web interface, but training and deployment of models will be done using the Python SDK. The SDK is available for Python 3.6 and later versions.


### VSCode integrated with Azure ML

To seamlessly work with Azure ML, you can use Visual Studio Code (VSCode) with the `Azure Machine Learning` extension. This extension provides a rich set of features for managing Azure ML resources, running experiments, and deploying models directly from your development environment.

Other handy extensions to install in VSCode are:

- Azure Storage - This extension allows you to manage Azure Storage resources, including Blob Storage, File Shares, and Queues. It provides a user-friendly interface for uploading, downloading, and managing files in your Azure Storage account.
- Azure Account - This extension provides a single sign-on experience for Azure services in VSCode. It allows you to log in to your Azure account and manage your Azure resources directly from the VSCode interface.
- Azure CLI Tools - This extension provides a set of commands for managing Azure resources directly from the VSCode terminal. It allows you to run Azure CLI commands without leaving the VSCode environment.
- Azure Resources - This extension provides a tree view of your Azure resources, allowing you to easily navigate and manage your Azure resources directly from the VSCode interface. It provides a user-friendly interface for creating, deleting, and managing Azure resources.
- Azure tools - This extension provides a set of tools for managing Azure resources, including Azure Functions, Azure Logic Apps, and Azure App Service. It allows you to create, deploy, and manage Azure resources directly from the VSCode environment.



## Azure Assets and Resources

Azure `assets` and `resources` are fundamental components required to build, deploy, and manage machine learning models in Azure ML. These include:

**Resources**: setup or infrastructural resources needed to run a machine learning workflow. Resources include:

- **[Workspace](#workspace)**: A centralized place to store and manage machine learning assets and resources.
- **[Compute Resources](#compute-instance)**: Virtual machines or clusters used to run training jobs, experiments, and deployments. Examples include Azure ML Compute Instances and Compute Clusters.
- **[Datastore](#datastore---including-how-to-reference-data-in-a-datastore)**: A centralized place to store the training data. Collections of data used for training and evaluating machine learning models. Data can be stored in various formats and locations, such as Azure Blob Storage, Azure Data Lake, or Azure SQL Database.

Resources need to be created within a `Resource group`. A resource group is a logical container that holds related Azure resources. It allows you to manage and organize your resources based on your project or application needs. You can create a resource group in the Azure portal or using Azure CLI. When creating a resource group, you need to specify a name and a region where the resources will be located. The region determines the physical location of the resources and can affect performance and cost. For the `RockJointDetection` project we have created a resource group called `rg-rock-joint-detection` in the `Norway East` region. You can check your resource groups in the Azure portal by clicking on `Resource groups` in the top menu.

**Assets**: created using Azure Machine Learning commands or as part of a training/scoring run. Assets are versioned and can be registered in the Azure Machine Learning workspace. They include:

- **[Models](#models)**: Serialized versions of trained machine learning models that can be registered, versioned, and deployed to endpoints.
- **[Environments](#environment)**: Configurations that define the software dependencies and runtime environment for training and inference. They ensure consistency and reproducibility of experiments.
- **[Data](#datastore---including-how-to-reference-data-in-a-datastore)**: For most use cases you refer to the data in the datastore in the form of a uri_folder or a uri_file. This is the data that is used for training and evaluation of the model.
- **[Experiments](#experiment-tracking-in-azure-ml)**: Collections of related training runs used to track and compare the performance of different models and configurations.

The steps below are not strictly necessary to train a model in Azure ML, but they are good practices to follow to ensure that your machine learning projects are well-organized, reproducible, and scalable. They are especially useful when working in a team or when managing multiple machine learning projects.

- **[Pipelines](#pipelines)**: Workflows that automate the process of training, evaluating, and deploying machine learning models. They can include multiple steps, such as data preprocessing, model training, and model evaluation. The pipelines include a number of Components that are executed in a sequence.
- **[Components](#component)**: Reusable building blocks that define a step in a pipeline. Components can be used to encapsulate code, data, and dependencies for a specific task, such as data preprocessing or model training. Think of them like functions.
- **[Endpoints](#endpoints)**: RESTful services that host deployed models for real-time scoring and batch inference. They provide a way to integrate machine learning models into applications.

These assets and resources are managed within the Azure ML workspace, providing a centralized platform for collaboration and management of machine learning projects.

## ML project lifecycle in Azure ML

A `workspace` organizes a project and allows for collaboration for many users all working toward a common objective. Users in a workspace can easily share the results of their runs from `experimentation` in the studio user interface. Or they can use versioned assets for jobs like `environments` and `storage` references.

When a project is ready for operationalization, users' work can be automated in an ML `pipeline` and triggered on a schedule or `HTTPS request`.

You can `deploy models` to the managed inferencing solution, for both real-time and batch deployments, abstracting away the infrastructure management typically required for deploying models.

The following diagram illustrates the ML project lifecycle in Azure ML:
![ML project lifecycle](images_documentation/overview-ml-development-lifecycle.png)

The ML model lifecycle is defined in the graphic below:
![ML model lifecycle](images_documentation/model-lifecycle.png)

### Organising and structuring the different resources across ML projects

Recommended structure for most users:

🔹 One resource group per environment or team
Example: ml-dev, ml-prod, ml-research

Easier to manage access, billing, and lifecycle.

🔹 One Azure ML workspace per environment
Example: ml-ws-dev, ml-ws-prod

Helps keep things organised between development and production.

🔹 Shared storage account and compute within workspace
Reuse compute clusters and environments across projects.

Use folder structures or naming conventions to separate projects in the storage.

🔹 Separation by conventions rather than infrastructure
Use naming conventions for:

Models: project1_model_a

Pipelines: proj2_data_cleaning_pipeline

Experiments: proj3_experiment_xyz

Tag assets by project using metadata or tags.

### Naming conventions

Use consistent, short, lowercase names with hyphens or underscores. Prefix names with the project identifier (`proj1`, `proj2`, etc.) to keep things organised across multiple projects.

### 🔁 General Rules
- Use only lowercase letters, numbers, hyphens (`-`), or underscores (`_`) where allowed.
- Keep names short but clear.
- Use version suffixes where needed (e.g. `_v1`, `_v2`).
- Add tags to help with filtering, cost tracking, and management.

---

### 🔧 Azure Resources

| Resource Type        | Naming Convention         | Example               |
|----------------------|---------------------------|-----------------------|
| Resource group       | `rg-ml-main`              | `rg-ml-main`          |
| ML workspace         | `ml-ws-main`              | `ml-ws-main`          |
| Storage account      | `st<project><suffix>`     | `stmlmain`            |
| Key vault            | `kv-<project>-<env>`      | `kv-ml-main`          |
| Container registry   | `acr<project>`            | `acrmlmain`           |

> Note: Storage account names must be globally unique, ≤ 24 chars, no hyphens or underscores.

---

### 📁 Storage Containers (in Blob)

| Type        | Naming Convention         | Example            |
|-------------|---------------------------|--------------------|
| Raw data    | `proj1-data`              | `proj1-data`       |
| Processed   | `proj2-outputs`           | `proj2-outputs`    |
| Models      | `proj2-models`            | `proj2-models`     |

---

### 🧠 ML Assets

| Asset Type   | Naming Convention             | Example                     |
|--------------|-------------------------------|-----------------------------|
| Experiment   | `proj1_exp_<task>`            | `proj1_exp_training`        |
| Dataset      | `proj1_dataset_<type>`        | `proj1_dataset_cleaned`     |
| Model        | `proj2_model_<algo>_v<ver>`   | `proj2_model_rf_v1`         |
| Pipeline     | `proj1_pipeline_<stage>`      | `proj1_pipeline_training`   |
| Environment  | `env-<project>-<lib>`         | `env-proj1-torch112`        |

---

### 🖥️ Compute

| Type            | Naming Convention           | Example              |
|-----------------|-----------------------------|----------------------|
| CPU cluster     | `cpu-cluster-<scope>`       | `cpu-cluster-general`|
| GPU cluster     | `gpu-cluster-<project>`     | `gpu-cluster-proj2`  |
| Inference       | `infer-cluster-<project>`   | `infer-cluster-proj1`|

---

### 🔐 Secrets (in Key Vault)

| Secret         | Naming Convention              | Example                   |
|----------------|--------------------------------|---------------------------|
| API keys       | `proj2-api-key`                | `proj2-api-key`           |
| DB connection  | `proj1-database-conn`          | `proj1-database-conn`     |

---

### 🏷️ Recommended Tags

Apply these tags across all resources:

```json
{
  "project": "proj1",
  "env": "dev",
  "owner": "yourname",
  "purpose": "training"
}
```


### Workspace

For machine learning teams, the workspace is a place to organize their work. Here are some of the tasks you can start from a workspace:

- Create jobs - Jobs are training runs you use to build your models. You can group jobs into experiments to compare metrics. E.g. by harnessing the Mlflow integration.
- Author pipelines - Pipelines are reusable workflows for training and retraining your model.
- Register data assets - Data assets aid in management of the data you use for model training and pipeline creation.
- Register models - Once you have a model you want to deploy, you create a registered model.
- Create online endpoints - Use a registered model and a scoring script to create an online endpoint.

#### Organising workspaces:

- One IT-admin user for all workspaces
- At least one admin role for each project. This role is responsible for managing the workspace, including creating compute instances, datastores, and managing access to the workspace.
- Create one workspace for each project. While a workspace can be used for multiple projects, limiting it to one project per workspace allows for cost reporting accrued to a project level. It also allows you to manage configurations like datastores in the scope of each project.
- Share Azure resources between workspaces, such as compute instances and datastores, to reduce costs.
- Share assets between workspaces, such as datasets and models, to reduce duplication of work.

#### Setting up a workspace:

1. Make sure you have a [Microsoft Azure account/subscription](#azure-accountsubscription)
2. Log into Azure ML (http://ml.azure.com) and choose workspaces
3. Create a new workspace by clicking: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?view=azureml-api-2&tabs=azure-portal

#### Content stored in a workspace:

Your workspace keeps a history of all training runs, with logs, metrics, output, lineage metadata, and a `snapshot of your scripts`. As you perform tasks in Azure Machine Learning, artifacts are generated. Their metadata and data are stored in the workspace and on its associated resources.

#### Connect to a workspace:

To connect to a workspace, you need to know the following information:

- The name of the workspace
- The resource group in which the workspace is located
- The subscription ID

Connect to the workspace in Python using that information.

First install the necessary dependencies:

```bash
poetry add azure-ai-ml azure-identity
```

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential

# Connect to Azure ML workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="your-subscription-id",
    resource_group_name="your-resource-group",
    workspace_name="your-workspace-name"
)

```

### Compute instance

Azure ML compute instances are virtual machines that you can use to run your training scripts. They are fully managed and can be used for training, inferencing, and data processing. Compute instances can be used for interactive development, training, and deployment of machine learning models. You define a compute instance in your workspace and use it to run your training scripts.

To create a new GPU cluster in Azure ML follow these steps:

1. Log into Azure ML and choose workspaces
2. Choose the workspace you want to create the cluster in. Note: the workspace need to be in a region that supports the GPU clusters you want. For example, `North Europe` does not support GPU clusters. Europe North support several clusters.
3. Click on `Compute` in the left menu
4. Click on `Compute instances` tab and choose `+ New` to create a new compute instance
5. Fill in the details for the cluster, such as name, type, and size.
6. Click `Create` to create the cluster


Using azure CLI

```bash
az ml compute create --name gpu-cluster-ncas8lowcost --type AmlCompute --size Standard_NC8as_T4_v3 --min-instances 0 --max-instances 2 --idle-time-before-scale-down 300 --resource-group rg-rock-joint-detection --workspace-name ws-rock-joint-det
```



Here is an overview of available compute types in Azure ML relevant for ML training:

| **GPU Type**       | **Compute Series**  | **Use Case**                  | **Performance Notes**                          | **GPU Count** | **Recommendation**                   |
|--------------------|---------------------|-------------------------------|------------------------------------------------|---------------|--------------------------------------|
| **NVIDIA T4**      | NCas_T4_v3          | Inference, lightweight training | Cost-effective, low power, supports mixed precision | 1             | ✅ *Best for inference* and testing  |
| **NVIDIA V100**    | NCv3                | Training mid-size models       | Strong FP32/FP16 performance, good memory      | 1–4           | ✅ *Best balance of speed and cost*  |
| **NVIDIA A100**    | NDv5                | Large-scale model training     | Very high performance, high memory and bandwidth | 1–8           | 🔥 *For large models and fast training* |

---

Choosing the Right Node:

- **🟢 Training U-Net or similar deep learning models on medium datasets (e.g. ~1000 images, 800×800):**
  Use **V100 (NCv3)** – it offers a good balance between training speed, memory, and cost. Suitable for most research and development tasks. Typical training cost: €2.50–€3.50 per hour. Example: 10 hours of training with NC6s_v3 (110 GB RAM) will cost ~ 36$ in North Europe. This is pricing in the "Pay-as-you-go" model. The cost may vary depending on the region and the type of subscription you have.

- **🔵 When cost is a concern, or for early experimentation:**
  Use **T4 (NCas_T4_v3)** – lower cost, less memory, slower training, but great for early-stage experiments or smaller batch sizes. Cost-effective at ~€0.30–€0.45 per hour. Example: 10 hours of training with NC8as_T4_v3 (56 GB RAM) will cost ~ 11$ in North Europe.
  → **Also the best choice for deploying models for inference.**

- **🔴 For high-end training needs (large datasets, large models, or faster results):**
  Use **A100 (NDv5)** – more expensive but excellent for resource-intensive training. Ideal when training time matters or for very large models. Costs from €4.50+ per hour per GPU

### Datastore - including how to reference data in a datastore

For efficient training of models, it is important to store the training data in a centralized location. **Azure blob storage** is a good place to store the training data. The data can be accessed by the training script running on the Azure ML compute instance.

Advantages of Using Azure Blob Storage and ML:

- **Scalability**: Easily handle large datasets without worrying about local storage limitations.
- **Data Proximity**: Keep your data close to the compute resources to reduce latency and improve performance.
- **Reproducibility**: Azure ML datasets are versioned, ensuring consistent data access across runs.
- **Cost Efficiency**: Use Azure Blob Storage tiers (e.g., hot, cool) to optimize storage costs.
- **Simplified Workflow**: Azure ML provides seamless integration with datasets, environments, and compute clusters.

Other datastorage alternatives are:

- Azure Data Lake Storage - A scalable and secure data lake for big data analytics.
- Azure SQL Database - A fully managed relational database service.
- Azure PostgreSQL - A fully managed open-source database service.
- Azure Cosmos DB - A globally distributed, multi-model database service.


#### 1. Prepare and upload your dataset to Azure blob storage

Ensure your dataset is structured and ready for machine learning. For example, you might organize the images into separate folders by class or group them based on their purpose (training, validation, testing).

Azure Blob Storage is a cost-effective, scalable storage solution. Use the Azure CLI or Python SDK to upload your dataset.

```bash
# Install Azure CLI if not already installed
az account set --subscription "your-subscription-name-or-id"
```


```bash
az storage blob upload-batch \
    --account-name <your-storage-account-name> \
    --container-name <your-container-name> \
    --source <local-folder-path>
```

```bash
az storage blob upload-batch \
    --account-name <your-storage-account-name> \
    --container-name azureml-blobstore-d3360a09-af14-4d39-8db4-b77f3097eaf3 \
    --source <local-folder-path>
```



or with Python SDK

```python
import os
local_folder = "./dataset"
for file_name in os.listdir(local_folder):
    blob_client = container_client.get_blob_client(file_name)
    with open(os.path.join(local_folder, file_name), "rb") as data:
        blob_client.upload_blob(data)
```


#### 2. Register the dataset

```python
dataset = Data(
    name="rock-segmentation-dataset",
    version="1",  # Increment this for new versions
    description="Dataset for rock segmentation with annotated joint lines",
    path="azureml://datastores/workspaceblobstore/paths/dataset-folder",
    type="uri_folder",
)

ml_client.data.create_or_update(dataset)

```

### Environment

Azure Machine Learning environments are an encapsulation of the environment where your machine learning task happens. They specify the software packages, environment variables, and software settings around your training and scoring scripts. The environments are managed and versioned entities within your Machine Learning workspace. Environments enable reproducible, auditable, and portable machine learning workflows across various computes.

1. Make a `environment.yml` listing all your dependencies from your `poetry` environment by first exporting to a `requirements.txt` file and then converting it to a `environment.yml` file (paste the content of the `requirements.txt` file into the `environment.yml` file).

**NOTE**: for Poetry version 2.0.0 and above export functionality need to be installed separately. Install it by running:

```bash
poetry self add poetry-plugin-export
```

Then export the dependencies to a `requirements.txt` file:


```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```


Here is an example output yaml file.

```yaml
name: rock-segmentation-env
channels:
  - defaults
dependencies:
  - python=3.9
  - pip
  - pip:
      - torch
      - torchvision
      - azure-ai-ml
      - azure-identity
      - your-other-packages
```

2. Register the environment in Azure ML:

```python
from azure.ai.ml.entities import Environment

env = Environment(
    name="rock-segmentation-env",
    description="Environment for rock segmentation training",
    conda_file="./environment.yml"
)
ml_client.environments.create_or_update(env)
```

Reference: https://learn.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-v2?view=azureml-api-2&tabs=sdk#environment

### Models

Azure Machine Learning models consist of one or more binary files that represent a machine learning model and any corresponding metadata. Models can be created from a local or remote file or directory. For remote locations `https, wasbs and azureml` locations are supported. The created model is tracked in the workspace under the specified name and version. Azure Machine Learning supports three types of storage format for models:

- custom_model
- mlflow_model
- triton_model

In NGI we try to standardise on the `mlflow_model` format. This is because it is a standard format that can be used across different platforms and tools. The `mlflow_model` format is a directory containing the model artifacts and a `MLmodel` file that describes the model. The model artifacts can be any file format, such as a pickle file, a TensorFlow SavedModel, or an ONNX model.


Reference: https://learn.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-v2?view=azureml-api-2&tabs=sdk#model

### Pipelines

### Component

An Azure Machine Learning component is a self-contained piece of code that does one step in a machine learning pipeline. Components are the building blocks of advanced machine learning pipelines. Components can do tasks such as data processing, model training, model scoring, and so on. A component is analogous to a function - it has a name, parameters, expects input, and returns output.

**Why should I use a component?**
It's a good engineering practice to build a machine learning pipeline to split a complete machine learning task into a multi-step workflow. Such that, everyone can work on the specific step independently. In Azure Machine Learning, a component represents one reusable step in a pipeline. Components are designed to help improve the productivity of pipeline building. Specifically, components offer:

Well-defined interface: Components require a well-defined interface (input and output). The interface allows the user to build steps and connect steps easily. The interface also hides the complex logic of a step and removes the burden of understanding how the step is implemented.

Share and reuse: As the building blocks of a pipeline, components can be easily shared and reused across pipelines, workspaces, and subscriptions. Components built by one team can be discovered and used by another team.

Version control: Components are versioned. The component producers can keep improving components and publish new versions. Consumers can use specific component versions in their pipelines. This gives them compatibility and reproducibility.

Unit testable: A component is a self-contained piece of code. It's easy to write unit test for a component.

Reference: https://learn.microsoft.com/en-us/azure/machine-learning/concept-component?view=azureml-api-2

## Training a model in Azure ML

When you run the script locally, Azure ML SDK connects your local machine to the Azure Machine Learning workspace. The script will then upload the training script and the training data to the Azure ML workspace and run the training job on the Azure ML compute instance.

1. Set up the Azure ML workspace
2. Register the dataset
3. Set up the coding environment
4. Define the `job script`.

Example job script to execute a trainig session. Notice that your original training script is being referenced in defining the job. Lets call it `submit_job.py`.

```python

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Command
from azure.identity import DefaultAzureCredential

# Connect to your Azure ML workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="your-subscription-id",
    resource_group_name="your-resource-group",
    workspace_name="your-workspace-name"
)

# Define the compute cluster name you have defined in your workspace
compute_cluster_name = "gpu-cluster"

# Define the command job
job = Command(
    code="./",  # Path to your repo
    command="python scripts/train.py",
    environment="./environment.yml",
    compute=compute_cluster_name,
    display_name="rock-segmentation-job",
    experiment_name="rock-segmentation",
)

# Submit the job
ml_client.jobs.create_or_update(job)
# Stream the terminal outputs locally
ml_client.jobs.stream(job.name)
```

5. Submit the job by running the script in your local environment:

```bash
python submit_job.py
```

6. Monitor the job via Azure portal or Azure CLI:

```bash
az ml job list --workspace-name your-workspace-name
```

7. Once the job is complete you can download the results from the Azure portal or using the Python SDK:

```python
job_details = ml_client.jobs.get(name="rock-segmentation-job")
ml_client.jobs.download(name=job_details.name, output_path="./outputs")
```

### Typical workflow for training a model in Azure ML

1. Develop code locally. Run your `scripts/train_eval.py` until you are sure that the scripts works. Local development is more efficient, and less costly than running in the cloud.
2. Run one experiment in Azure and check that results make sense
3. Run several experiments, hyperparameter optimisation etc in Azure
4. Choose a model for results and eventually for deployment


### Complete script for training a model in Azure ML

Add this in your repo and version control it. This script will be used to train the model in Azure ML. For local prototyping you dont need it.

```python

```




## Experiment tracking in Azure ML

### MLflow

Azure ML supports integration with MLflow for tracking experiments. You can use MLflow to log parameters, metrics, and artifacts in your `train.py` script. Azure ML automatically sets up MLflow for tracking. You dont need to add a URI to the MLflow server.

Example code to add in your `train.py` script to enable MLflow tracking:

```python
import mlflow
import mlflow.pytorch

# Start MLflow experiment
mlflow.start_run()

# Log parameters
mlflow.log_param("learning_rate", 0.01)

# Log metrics
mlflow.log_metric("accuracy", 0.95)

# Log model
model = ...  # Your trained model
mlflow.pytorch.log_model(model, "models/rock_segmentation") # pytorch example
mlflow.scikit.log_model(model, "models/rock_segmentation") # scikit example

# End MLflow run
mlflow.end_run()
```

Logs are available in the Azure ML Studio under the experiment details. You can also fetch logs programmatically:

```python
job = ml_client.jobs.get("job-name")
print(job.outputs)
```

The logs will be saved in Azure ML to the `outputs/mlruns` folder.


### Tensorboard

Training deep learning models often takes time. To log the progress of metrics and loss during training, you can use TensorBoard. Azure ML provides integration with TensorBoard for visualizing training metrics.

Example code to add in your `train.py` script to enable TensorBoard logging:

```python
from torch.utils.tensorboard import SummaryWriter
import torch
import time

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="./outputs/tensorboard_logs")

# Simulated training loop
for epoch in range(10):
    loss = torch.rand(1).item()  # Example loss value
    accuracy = torch.rand(1).item()  # Example accuracy value

    # Log metrics
    writer.add_scalar("Loss/train", loss, epoch)
    writer.add_scalar("Accuracy/train", accuracy, epoch)

    time.sleep(1)  # Simulate time between epochs

# Close the writer
writer.close()
```

The logs will be saved in Azure ML to the `outputs/tensorboard_logs` folder. You can view the TensorBoard logs in Azure ML Studio under the experiment details.

Once the training job is running or completed you can view the TensorBoard logs in the Azure ML Studio under the experiment details for the specific run.

## Dataset versioning in Azure ML

Azure ML provides robust support for dataset versioning to ensure that the datasets used for machine learning experiments are version-controlled.

### Register a Dataset in Azure ML

When you register a dataset in Azure ML, it is automatically versioned. Each time you register a dataset with the same name but with different content, a new version is created.

#### Example: Register a Dataset

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential

# Connect to Azure ML workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="your-subscription-id",
    resource_group_name="your-resource-group",
    workspace_name="your-workspace-name"
)

# Register the dataset
dataset = Data(
    name="rock-segmentation-dataset",
    version="1",  # Increment this for new versions
    description="Dataset for rock segmentation with annotated joint lines",
    path="azureml://datastores/workspaceblobstore/paths/dataset-folder",
    type="uri_folder",
)

ml_client.data.create_or_update(dataset)
```

---

### View Dataset Versions
You can query and list all versions of a dataset.

#### List All Versions of a Dataset
```python
datasets = ml_client.data.list(name="rock-segmentation-dataset")
for dataset in datasets:
    print(f"Name: {dataset.name}, Version: {dataset.version}, Path: {dataset.path}")
```

This helps you track changes and identify the dataset version used in an experiment.

---

### Use a Specific Dataset Version in an Experiment
When defining an experiment, you can specify the dataset version explicitly. This ensures that the experiment always uses the intended version of the dataset.

#### Example: Use a Specific Dataset Version
```python
from azure.ai.ml.entities import Command

job = Command(
    code="./",  # Path to your training script
    command="python scripts/train.py --data-dir azureml:/data/rock-segmentation-dataset@1",
    environment="rock-segmentation-env:latest",
    compute="gpu-cluster",
    display_name="rock-segmentation-training",
    experiment_name="rock-segmentation-experiment",
)
ml_client.jobs.create_or_update(job)
```

The `@1` in `azureml:/data/rock-segmentation-dataset@1` specifies the version of the dataset.

---

### Dataset Versioning in the Experiment Log
Azure ML automatically logs the dataset version used in each experiment. You can view this in Azure ML Studio or programmatically.

#### Retrieve Dataset Information from a Job
```python
job = ml_client.jobs.get("job-name")
print(job.inputs["data-dir"].path)  # Outputs the dataset path with version
```

---

### 5. Benefits of Dataset Versioning
- **Reproducibility**: You can re-run experiments with the exact dataset version used originally.
- **Traceability**: Experiment logs include dataset version details.
- **Collaboration**: Teams can access shared, versioned datasets across experiments.
- **Compliance**: Meets regulatory requirements for tracking data used in model training.

---

### 6. Updating Datasets
To update an existing dataset with new data, re-register it with the same name but a new version. Azure ML retains all previous versions for reference.

#### Register a New Version
```python
dataset = Data(
    name="rock-segmentation-dataset",
    version="2",
    description="Updated dataset with additional samples",
    path="azureml://datastores/workspaceblobstore/paths/dataset-folder-updated",
    type="uri_folder",
)

ml_client.data.create_or_update(dataset)
```

---

### 7. Data Asset Management in Azure ML Studio
You can also manage dataset versions via the Azure ML Studio:

1. Navigate to Datasets in the Studio.
2. Select a dataset to view all versions.
3. View metadata, paths, and consumption logs for each version.

---

### 8. Integrating with Git and CI/CD

For comprehensive version control:

- Store dataset definitions (e.g., metadata, paths) in Git.
- Use CI/CD pipelines to register datasets programmatically.
- Reference dataset versions in model deployment pipelines.

---

By leveraging Azure ML’s dataset versioning, you create a reproducible, traceable, and scalable workflow for managing data assets in your machine learning projects.


## AutoML

A step in the prototyping phase. More...

## Example tutorials

We have included one beginners tutorial and two example projects from NGI where we demonstrate the functionality in Azure ML.

### Predicting Titanic survival

TODO: make a repo for this tutorial. Adapt the code to tools and principles used in NGI.

The tutorial handles Azure ML concepts like compute, environment, data asset, tracking, model storage. Functionality is mainly demonstrated in a jupyter notebook.

https://medium.com/henkel-data-and-analytics/how-to-use-azure-ml-studio-an-eye-opening-model-training-tutorial-for-beginners-from-henkels-data-5035ee10a6d2

![Titanic example](example_tutorial_titanic.webp)

### Rock joint detection

coming...

### LabOAI

coming...

## Learning resources

- https://learn.microsoft.com/en-us/training/paths/explore-azure-machine-learning-workspace/
- https://learn.microsoft.com/en-us/training/paths/train-deploy-machine-learning-model/
