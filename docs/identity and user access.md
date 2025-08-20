# User access, permissions and identity management in Azure ML

This section summarises essential identity and access concepts for working with Azure Machine Learning (AML) within your organisation. It focuses on best practices and actionable steps, avoiding unnecessary complexity.

## Core Concepts

### Microsoft Entra ID (formerly Azure Active Directory)

Microsoft Entra ID is the identity and access management system used by Azure. All users, service principals, and managed identities are registered here.

### Azure Role-Based Access Control (RBAC)

RBAC controls access to Azure resources. Access is granted by assigning roles to users, groups, or identities at different scopes: subscription, resource group, or individual resource.

### Roles

Typical roles in RBAC include:

* **Owner**: Full control, including managing access.
* **Contributor**: Full control except access management.
* **Reader**: View-only access.
* **ML-specific roles (examples)**:

  * `AzureML Data Scientist`
  * `AzureML Contributor`
  * `Storage Blob Data Reader` (for accessing blob storage)

### Credentials and Tokens

* **Credential**: A credential is the means by which a user or identity proves who they are to a system. Examples include username/password, certificates, or a managed identity. Credentials are used during the authentication step.

* **Token**: A token is a short-lived, digitally signed piece of data issued after successful authentication using a credential. It acts as a key that grants access to specific resources and is passed between services during API calls. Tokens are issued by Microsoft Entra ID or Azure ML and typically expire within a short period (e.g., one hour).

* **Difference**: Credentials are secret and long-term (e.g. a password or identity), while tokens are temporary and scoped to particular operations. You should avoid using credentials directly in scripts and instead rely on tokens generated via secure identity mechanisms like `DefaultAzureCredential`.

* **Tenant**: Represents your organisation in Microsoft Entra ID. It defines a security boundary and holds all users, applications, and identities for your organisation. Identified by a tenant ID.

## Identity Types

An **identity** in Azure represents a security principal — a user, group, service, or resource that needs to authenticate and access Azure resources. An identity is what Azure uses to determine **who you are**.

Identities are authenticated using **credentials** (e.g. passwords, certificates, managed identities). Once authenticated, a **token** is issued, which is then used to authorise access to resources. This separation helps increase security and flexibility in managing access.

### User Identity

Used when you log in via Azure CLI, SDK, or portal. Example: your NGI user account.

### Group Identity

Recommended for managing access at scale. Permissions assigned to groups automatically apply to all group members.

### Managed Identity

Managed identities provide secure, automatic authentication for Azure resources without requiring you to manage credentials. This is especially useful in ML pipelines, where compute resources need access to data without manual token handling. For example, when a compute cluster needs to read data from Azure Blob Storage, it can use a managed identity to authenticate without hardcoding credentials.

### Benefits compared to alternatives:

* **Credentialless authentication**: No hardcoded secrets or credentials needed.
* **Automated lifecycle**: System-assigned identities are automatically managed by Azure.
* **Secure by default**: Limited to only the permissions assigned via RBAC.
* **Supports least-privilege principle**: You assign only the minimum roles needed.
* **Scales well**: Especially useful when using multiple compute instances, pipelines, or automated jobs.
* **Avoids dependency on Azure Key Vault**: While Azure Key Vault is a secure way to store secrets like passwords and connection strings, using managed identity eliminates the need for storing secrets at all. This reduces complexity and the risk of leaked credentials.

You are moving away from the alternativ process of:
1. Create service principal
2. Grant permissions
3. Set credentials in environment variables or config files
4. Store credentials/secrets in Azure Key Vault (on resource group or workspace level)
5. Rotate secrets periodically
6. Clean up secrets when no longer needed
7. Delete service principal when no longer needed

To a process that uses managed identity:
1. Create Azure resource (e.g. compute) with managed identity
2. Grant permissions to that identity
3. Delete resource when no longer needed (identity is cleaned up automatically)

### Why managed identity is preferred over Azure Key Vault for AML jobs:

* **Key Vault requires secrets to be created, stored, and rotated**. With managed identities, these concerns are handled automatically by Azure.
* **Simpler integration**: Managed identity works natively with Azure services like blob storage, AML, and compute resources without additional setup.
* **No manual secret injection**: You don’t need to load secrets into environment variables or mount volumes in compute nodes.
* **BETTER SECURITY HYGIENE**: Credentials never appear in code or config files, reducing the risk of accidental exposure. Thus, managed identities leads to sanitising code so credentials dont get exposed in logs, scripts, or version control.

### Identity types:

* **System-assigned**: Tied to a specific resource, like a compute cluster. Automatically created and deleted with the resource. Recommended for most AML compute scenarios (training and inference), since it ensures each compute resource has its own identity.
* **User-assigned**: A standalone identity that can be used by multiple resources. Recommended when you want to:

  * Share identity across compute resources
  * Reuse across workspaces or resource groups
  * Maintain consistent identity even if compute is deleted

For most training and inference jobs, **system-assigned managed identity** is preferred due to simplicity and tighter scoping. Use **user-assigned identity** when you need more flexibility or cross-resource reuse.

### Identity types:

* **System-assigned**: Tied to a specific resource, like a compute cluster. Automatically created and deleted with the resource. Recommended for most AML compute scenarios (training and inference), since it ensures each compute resource has its own identity.
* **User-assigned**: A standalone identity that can be used by multiple resources. Recommended when you want to:

  * Share identity across compute resources
  * Reuse across workspaces or resource groups
  * Maintain consistent identity even if compute is deleted

For most training and inference jobs, **system-assigned managed identity** is preferred due to simplicity and tighter scoping. Use **user-assigned identity** when you need more flexibility or cross-resource reuse.


### DefaultAzureCredential

A unified way to authenticate from Python code. It automatically selects the right identity:

1. Environment variables
2. Managed identity
3. Azure CLI credentials (your user)

Example:

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Connect to Azure ML workspace
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="your-subscription-id",
    resource_group_name="your-resource-group",
    workspace_name="your-workspace-name"
)
```

## Access Inheritance and Scopes

Permissions are inherited:

* **Subscription** > **Resource Group** > **Resource**
  Assigning roles at a higher level (e.g. resource group) makes management easier but less granular.

## Best Practices for Access Management

### Recommended Role for Developers

For users developing, training, and managing machine learning models in Azure ML, the most appropriate role is **AzureML Data Scientist**. This role provides permissions needed to:

- Run experiments and jobs
- Register, deploy, and manage models
- Access datasets and datastores linked to the workspace

It does not include permissions for workspace-wide configuration or user management, making it suitable for regular developers.

If broader access is needed (e.g. to configure compute or modify workspace settings), the **AzureML Contributor** role may be appropriate, but this should be limited to senior team members due to the extended permissions it grants.

### Granting Access

Use **resource group-level** role assignments for most use cases. It simplifies management and reduces permission errors. Assigning roles at the subscription level grants access to all current and future resources under that subscription, which can be risky and overly permissive. In contrast, resource group-level assignments offer a more controlled and least-privilege approach, letting you limit access to only the necessary set of resources. This also improves auditability and reduces the chance of accidental changes to unrelated resources.

To add users:

1. Go to the [Azure portal](https://portal.azure.com/)
2. Find the resource group or workspace
3. Select **Access control (IAM)**
4. Add role assignment (e.g., Contributor, Reader, AzureML Data Scientist)

## Managing Access: Azure CLI and Portal

### Azure CLI

```bash
# Log in with user identity
az login

# (Optional) Set subscription
az account set --subscription <your-subscription-id>
# example
az account set --subscription ngi_mlops_sandbox

# Assign a role to a user at the resource group level
az role assignment create \
  --assignee <user-email-or-object-id> \
  --role "AzureML Contributor" \
  --scope "/subscriptions/<sub-id>/resourceGroups/<rg-name>"
```

### Portal Interface

1. Navigate to your **Resource Group** or **AML Workspace**.
2. Click **Access control (IAM)**.
3. Click **Add > Add role assignment**.
4. Select role and user.


## Setting Up Managed Identity for Compute (Step-by-Step)

To ensure a smooth and secure setup for ML training and inference jobs using managed identity, follow this structured approach:

### 1. Create or Configure Compute with System-Assigned Identity

When creating a compute cluster (e.g. with `az ml compute create` or via the Azure ML Studio portal), enable the **system-assigned managed identity**:

#### Web Portal

* Navigate to your Azure ML workspace
* Go to **Compute > + New compute cluster**
* In the **Advanced settings**, enable **Assign system identity**

#### CLI

```bash
az ml compute create \
  --name my-compute \
  --type AmlCompute \
  --size Standard_DS3_v2 \
  --min-instances 0 \
  --max-instances 4 \
  --identity-type SystemAssigned
```

### 2. Grant Storage Access to Compute Identity

Once the compute cluster is created, its managed identity needs access to the data:

#### Determine the identity object ID:

```bash
az ml compute show \
  --name <compute-name> \
  --resource-group <rg-name> \
  --workspace-name <workspace-name> \
  --query identity.principal_id -o tsv

# Example
az ml compute show --name Standard-NC6s-v3 --resource-group rg-rock-joint-detection --workspace-name ws-rock-joint-det --query identity.principal_id -o tsv

# returns 6b64c8b6-82db-4894-b6ea-3d60dc2292de

```

#### Assign the role on storage account:

```bash
az role assignment create \
  --assignee <principal-id-from-above> \
  --role "Storage Blob Data Reader" \
  --scope "/subscriptions/<sub-id>/resourceGroups/<rg-name>/providers/Microsoft.Storage/storageAccounts/<storage-account-name>"

# Example
az role assignment create --assignee 6b64c8b6-82db-4894-b6ea-3d60dc2292de --role "Storage Blob Data Reader" --scope "/subscriptions/0edfb8db-083f-4467-a411-d497eaa6b41e/resourceGroups/rg-rock-joint-detection/providers/Microsoft.Storage/storageAccounts/strockjointdet"
```
This ensures compute can read from datastores backed by the blob storage when doing the machine learning training.

#### How to Validate Storage Access Permissions for Compute

##### Web Portal (GUI)


After assigning the managed identity of a compute cluster to the storage account, it’s good practice to validate that the permissions are correctly set.

##### Method A: Check Role Assignments in Azure Portal

You can verify that the role assignment has been applied using the Azure portal:

1. Go to the **Storage Account** in the Azure portal.
2. Navigate to **Access control (IAM)**.
3. Click **Role assignments**.
4. In the filter, enter the name or object ID of your compute’s managed identity.
5. Ensure that the role `Storage Blob Data Reader` (or equivalent) is listed.

This GUI-based check is useful for confirming correct configuration without using CLI commands.

##### Method B: Submit a Test Job that Accesses Data

* The most direct validation is to submit a small Azure ML job that attempts to read from the datastore (backed by the storage account).
* If the managed identity lacks the right permissions, the job will fail with clear errors such as:

  * `AuthorizationPermissionMismatch`
  * `NoIdentityOnCompute`

##### Method C: Use Azure CLI to List Role Assignments

You can confirm that the compute’s managed identity has the right role:

```bash
az role assignment list \
  --assignee <principal-id-of-compute> \
  --scope "/subscriptions/<sub-id>/resourceGroups/<rg-name>/providers/Microsoft.Storage/storageAccounts/<storage-account-name>"

# Example
az role assignment list --assignee 6b64c8b6-82db-4894-b6ea-3d60dc2292de --scope "subscriptions/0edfb8db-083f-4467-a411-d497eaa6b41e/resourceGroups/rg-rock-joint-detection/providers/Microsoft.Storage/storageAccounts/strockjointdet"

```

Check that the role includes `Storage Blob Data Reader` or another appropriate role.

##### Method D: Access from Python Code in a Job

In your training script, you can explicitly try to access a file using the Azure SDK:

```python
from azure.storage.blob import BlobServiceClient

account_url = "https://<storage-account-name>.blob.core.windows.net"
container_name = "my-container"
blob_name = "testfile.csv"

blob_service = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
container_client = blob_service.get_container_client(container_name)
blob_client = container_client.get_blob_client(blob_name)

# Try downloading the blob
with open("downloaded_test.csv", "wb") as f:
    f.write(blob_client.download_blob().readall())
```

If the managed identity has the right permissions, the file will be downloaded. Otherwise, an `HttpResponseError` will be raised indicating a permission issue.

These validation steps help confirm your configuration is complete and functioning before running full training workloads.



### 3. Reference Compute and Identity in Training Scripts

When you submit a job via Python SDK using `command()` from `azure.ai.ml`, no special credentials are needed if you rely on `DefaultAzureCredential`. AML will use the compute’s managed identity. Instead of passing a string to the `identity` parameter, it is recommended to use the `ManagedIdentityConfiguration` object from `azure.ai.ml.entities` for clarity and correctness.

```python
from azure.ai.ml import command
from azure.ai.ml.entities import ManagedIdentityConfiguration

training_job = command(
    code="./src",
    command="python train.py --data ${{inputs.data}}",
    inputs={"data": input_data},
    environment="azureml:my-env@latest",
    compute="my-compute",
    identity=ManagedIdentityConfiguration(type="SystemAssigned"),
    experiment_name="my-training-exp"
)
```

### 4. Validate Setup

Submit a test job that accesses a datastore (e.g. mounted blob storage). If configured correctly, the compute identity will seamlessly access the data. If not, typical errors like `NoIdentityOnCompute` or `AuthorizationPermissionMismatch` will occur (see Troubleshooting section).

### Summary of Tool Use

| Operation                    | Recommended Tool     |
| ---------------------------- | -------------------- |
| Create compute with identity | Azure CLI or Portal  |
| Enable identity on compute   | Azure CLI or Portal  |
| Grant RBAC role to identity  | Azure CLI preferred  |
| Reference identity in job    | SDK with `command()` |

This method ensures no hardcoded secrets are used, improves automation, and avoids access errors at runtime.


### 5. Additional Resources Requiring Managed Identity Configuration

In a typical machine learning project, in addition to compute clusters, other Azure resources or entities may also require managed identity configuration to ensure secure and automated access:

#### a) Azure ML Pipeline Jobs

If your ML pipeline includes components that run on different compute resources or access various datastores, each compute used in the pipeline should have a managed identity with the appropriate RBAC roles assigned (e.g., Storage Blob Data Reader).

#### b) Inference Endpoints

When deploying models for inference (real-time or batch):

* **Managed online endpoints** can be configured with a managed identity.
* This identity should be granted access to model storage, feature stores, or other Azure resources it interacts with.

Example using Azure CLI:

```bash
az ml online-endpoint create \
  --name my-endpoint \
  --identity-type SystemAssigned \
  --file endpoint.yaml
```

In your `endpoint.yaml`:

```yaml
identity:
  type: SystemAssigned
```

#### c) Azure Key Vault (if used)

If your project uses Azure Key Vault to manage secrets (e.g. API keys, database passwords), grant the managed identity of your compute or endpoint **Key Vault Reader** or **Secrets User** roles.

#### d) Azure Container Registry (ACR)

When using private Docker images from ACR, ensure that your compute’s managed identity is granted **AcrPull** role on the registry:

```bash
az role assignment create \
  --assignee <compute-principal-id> \
  --role "AcrPull" \
  --scope "/subscriptions/<sub-id>/resourceGroups/<rg-name>/providers/Microsoft.ContainerRegistry/registries/<acr-name>"
```

---

By properly assigning managed identity roles to these resources, you avoid manual credential management and ensure a secure, scalable, and automated ML workflow across training, deployment, and monitoring stages.

















## Troubleshooting: Common Errors

### `NoIdentityOnCompute`

Occurs when the compute node lacks a managed identity or that identity has no access to required resources (e.g. blob storage).

**Fix:**

* Ensure system-assigned identity is enabled for the compute.
* Assign required RBAC roles (e.g. `Storage Blob Data Reader`) on the target datastore.

### `AuthenticationFailed` or `AuthorizationPermissionMismatch`

Typically due to insufficient roles or assigning the role at the wrong scope.

**Fix:**

* Double-check the RBAC scope.
* Confirm the user or identity has the correct role.

## Summary: Recommended Setup

| Resource           | Identity Type            | Recommended Role                      |
| ------------------ | ------------------------ | ------------------------------------- |
| Azure ML Workspace | User/Group Identity      | AzureML Contributor / Reader          |
| Compute Cluster    | System-Assigned Identity | Storage Blob Data Reader (on storage) |
| Blob Storage       | Compute Managed Identity | Storage Blob Data Reader              |
| Scripts & SDKs     | DefaultAzureCredential   | Leverages CLI or managed identity     |

## References

* [RBAC and role assignment](https://learn.microsoft.com/en-us/azure/role-based-access-control/role-assignments-portal)
* [Azure CLI authentication](https://learn.microsoft.com/en-us/cli/azure/authenticate-azure-cli-interactively)
* [Azure ML authentication](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication)
* [Azure ML service authentication](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-identity-based-service-authentication)
* [Manage workspace roles](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-assign-roles)
