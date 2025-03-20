# Azure ML


## Table of Contents
- [Azure ML](#azure-ml)
- [Table of Contents](#table-of-contents)
- [Tools and system setup](#tools-and-system-setup)
    - [Azure account/subscription](#azure-accountsubscription)
    - [Azure CLI](#azure-cli)
    - [VSCode integrated with Azure ML](#vscode-integrated-with-azure-ml)
- [Azure Assets and Resources](#azure-assets-and-resources)
    - [Workspace](#workspace)
        - [Organising workspaces](#organising-workspaces)
        - [Setting up a workspace](#setting-up-a-workspace)
        - [Content stored in a workspace](#content-stored-in-a-workspace)
        - [Connect to a workspace](#connect-to-a-workspace)
    - [Compute instance](#compute-instance)
    - [Datastore](#datastore)
        - [Prepare and upload your dataset to Azure blob storage](#prepare-and-upload-your-dataset-to-azure-blob-storage)
        - [Register the dataset](#register-the-dataset)
    - [Environment](#environment)
    - [Models](#models)
    - [Pipelines](#pipelines)
    - [Component](#component)
- [Training a model in Azure ML](#training-a-model-in-azure-ml)
- [Experiment tracking in Azure ML](#experiment-tracking-in-azure-ml)
    - [MLflow](#mlflow)
    - [Tensorboard](#tensorboard)
- [Dataset versioning in Azure ML](#dataset-versioning-in-azure-ml)
- [AutoML](#automl)
- [Learning resources](#learning-resources)


---

Azure ML has a comprehensive documentation. In this documentation for a code academy course in MLOps and professional ML development in Azure ML we have tried to simplify the most important parts of Azure ML and to extract the parts which currently are most relevant for NGI.

Reference: https://learn.microsoft.com/en-us/azure/machine-learning/?view=azureml-api-2

Azure ML is a cloud-based service for creating, managing, and deploying machine learning models. It provides a centralized place for data scientists and developers to work with all the artifacts for machine learning, including datasets, training scripts, and trained models. Azure ML give make it easy to scale your ML training by harnessing powerful cloud compute resources, such as nodes with several GPU's with lots of memory. Below we have listed the parts in a typical MLOps worklfow that is facilitated by Azure ML.

- Experiment tracking to mlflow or tensorboard
- Model training
- Model deployment
- Monitoring of deployed models
- Retraining of deployed models

MLOps steps not covered by Azure ML can be found in the [MLOps](MLOps.md) document.


References:
- https://learn.microsoft.com/en-us/azure/machine-learning/?view=azureml-api-2
- https://medium.com/henkel-data-and-analytics/how-to-use-azure-ml-studio-an-eye-opening-model-training-tutorial-for-beginners-from-henkels-data-5035ee10a6d2

## Tools and system setup

### Azure account/subscription

To use Azure ML you need an Azure account. You can create a free account at https://go.microsoft.com/fwlink/?linkid=2227353&clcid=0x409&l=en-us&icid=azurefreeaccount. The free account gives you access to a limited set of Azure services for 12 months. You can use the free account to explore Azure ML and other Azure services. Once you have an Azure account, you can create an Azure Machine Learning workspace to start building, training, and deploying machine learning models.

Reference: https://azure.microsoft.com/en-us/pricing/purchase-options/azure-account?icid=azurefreeaccount


### Azure CLI

Azure CLI is a command-line tool that provides a set of commands for managing Azure resources. You can use Azure CLI to create and manage Azure resources, such as virtual machines, storage accounts, and Azure Machine Learning workspaces. Azure CLI is available for Windows, macOS, and Linux. You can install Azure CLI on your local machine or use the Azure Cloud Shell, which is a browser-based shell that comes pre-installed with Azure CLI.


Reference: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public


### VSCode integrated with Azure ML

coming...


## Azure Assets and Resources

Azure assets and resources are the fundamental components required to build, deploy, and manage machine learning models in Azure ML. These include:

**Resources**: setup or infrastructural resources needed to run a machine learning workflow. Resources include:

- **Workspace**: A centralized place to store and manage machine learning assets and resources.
- **Compute Resources**: Virtual machines or clusters used to run training jobs, experiments, and deployments. Examples include Azure ML Compute Instances and Compute Clusters.
- **Datastore**: A centralized place to store the training data. Collections of data used for training and evaluating machine learning models. They can be stored in various formats and locations, such as Azure Blob Storage or Azure Data Lake.

**Assets**: created using Azure Machine Learning commands or as part of a training/scoring run. Assets are versioned and can be registered in the Azure Machine Learning workspace. They include:

- **Models**: Serialized versions of trained machine learning models that can be registered, versioned, and deployed to endpoints.
- **Environments**: Configurations that define the software dependencies and runtime environment for training and inference. They ensure consistency and reproducibility of experiments.
- **Data**: For most usecases you refer to the data in the datastore in the form of a uri_folder or a uri_file. This is the data that is used for training and evaluation of the model.
- **Experiments**: Collections of related training runs used to track and compare the performance of different models and configurations.

The steps below are not strictly necessary to train a model in Azure ML, but they are good practices to follow to ensure that your machine learning projects are well-organized, reproducible, and scalable. They are especially useful when working in a team or when managing multiple machine learning projects.

- **Pipelines**: Workflows that automate the process of training, evaluating, and deploying machine learning models. They can include multiple steps, such as data preprocessing, model training, and model evaluation. The pipelines includes a number of Components that are executed in a sequence.
- **Components**: Reusable building blocks that define a step in a pipeline. Components can be used to encapsulate code, data, and dependencies for a specific task, such as data preprocessing or model training. Think of them like functions.
- **Endpoints**: RESTful services that host deployed models for real-time scoring and batch inference. They provide a way to integrate machine learning models into applications.

These assets and resources are managed within the Azure ML workspace, providing a centralized platform for collaboration and management of machine learning projects.


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
2. Log into Azure and choose workspaces
2. Create a new workspace by clicking

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

1. Log into Azure and choose workspaces
2. Choose the workspace you want to create the cluster in
3. Click on `Compute` in the left menu
4. Click on `Create` and choose `Compute cluster` tab
5. Fill in the details for the cluster, such as name, type, and size. Ensure the cluster is GPU-enabled (e.g., Standard_NC6, Standard_NC12, or similar VM types).
6. Click `Create` to create the cluster


Here is an overview of available compute types in Azure ML:


| **GPU Type**       | **Compute Instance Series** | **Ideal For**                      | **Key Features**                          | **GPU Count** |
|---------------------|-----------------------------|-------------------------------------|-------------------------------------------|---------------|
| NVIDIA T4          | NCas_T4_v3                 | Inference, lightweight training     | Cost-effective, FP32, INT8                | 1             |
| NVIDIA V100        | NCv3                       | Deep learning training              | High memory, FP32, FP16                   | 1-4           |
| NVIDIA A100        | NDv5                       | Large-scale distributed training    | Tensor cores, FP64, FP16                  | 1-8           |
| NVIDIA K80         | NCv2                       | Budget training                     | Older architecture                        | 1-2           |
| NVIDIA P40         | NCv1                       | Training, inferencing (mid-range)   | Moderate memory and speed                 | 1-2           |
| NVIDIA A40         | NCas_A40_v4                | Advanced rendering and AI workloads | High performance, flexibility             | 1-2           |
| NVIDIA A100        | NDv4                       | Deep learning, HPC workloads        | High GPU interconnect (NVLink), FP16, FP64 | 1-8           |
| No GPU (D16_v5)    | Dv5                        | General-purpose compute workloads   | CPU-based, scalable, cost-efficient       | N/A           |


GPU nodes are NC and ND series. NC series are optimized for training deep learning models, while ND series are optimized for large-scale distributed training. The number of GPUs per node can vary from 1 to 8, depending on the series and size of the VM. D series are CPU-based nodes that are cost-effective and scalable for general-purpose compute workloads.

For inference workloads, you can use the NCas_T4_v3 series, which is optimized for cost-effective inference and lightweight training.


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
az storage blob upload-batch \
    --account-name <your-storage-account-name> \
    --container-name <your-container-name> \
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

### 1. Register a Dataset in Azure ML
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

### 2. View Dataset Versions
You can query and list all versions of a dataset.

#### List All Versions of a Dataset
```python
datasets = ml_client.data.list(name="rock-segmentation-dataset")
for dataset in datasets:
    print(f"Name: {dataset.name}, Version: {dataset.version}, Path: {dataset.path}")
```

This helps you track changes and identify the dataset version used in an experiment.

---

### 3. Use a Specific Dataset Version in an Experiment
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

### 4. Dataset Versioning in the Experiment Log
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

## Learning resources

- https://learn.microsoft.com/en-us/training/paths/explore-azure-machine-learning-workspace/
- https://learn.microsoft.com/en-us/training/paths/train-deploy-machine-learning-model/
