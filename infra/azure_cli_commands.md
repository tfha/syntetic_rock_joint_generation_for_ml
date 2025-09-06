# Azure CLI Commands


## Grant AML compute access to Azure Blob Storage (RBAC)

This guide assigns a data-plane role (Reader or Contributor) to an Azure ML compute’s managed identity at the storage account or container scope.

### Prerequisites
- Azure CLI installed and signed in: `az login`
- Azure ML CLI extension: `az extension add -n ml`
- RBAC permission to assign roles (Owner or role with roleAssignments/write)

---

## 1) Collect the required IDs

Use PowerShell variables so the rest of the commands are simple.

```powershell
# Adjust if needed
$SUBSCRIPTION_ID = "<subscription-id>"
$RESOURCE_GROUP  = "<resource-group-name>"
$WORKSPACE_NAME  = "<aml-workspace-name>"
$STORAGE_ACCOUNT = "<storage-account-name>"
$COMPUTE_NAME    = "<aml-compute-name>"

# Example with your values:
# $SUBSCRIPTION_ID = "0edfb8db-083f-4467-a411-d497eaa6b41e"
# $RESOURCE_GROUP  = "rg-rock-joint-detection"
# $WORKSPACE_NAME  = "ws-rock-joint-det"
# $STORAGE_ACCOUNT = "strockjointdet"
# $COMPUTE_NAME    = "NC64as-T4-v3"
```

- Subscription ID
```powershell
az account show --query id -o tsv
# or if you have multiple, list and select:
# az account list -o table
az account set --subscription $SUBSCRIPTION_ID
```

- Storage account resource ID (scope)
```powershell
$storageScope = az storage account show `
  --name $STORAGE_ACCOUNT `
  --resource-group $RESOURCE_GROUP `
  --query id -o tsv
$storageScope
```

- Compute principal object ID (managed identity)
```powershell
$computeObjectId = az ml compute show `
  --name $COMPUTE_NAME `
  --workspace-name $WORKSPACE_NAME `
  --resource-group $RESOURCE_GROUP `
  --query identity.principalId -o tsv
$computeObjectId
```

If this comes back empty, enable the compute’s managed identity and retry:
```powershell
az ml compute update `
  --name $COMPUTE_NAME `
  --workspace-name $WORKSPACE_NAME `
  --resource-group $RESOURCE_GROUP `
  --identity-type SystemAssigned
# then re-run the 'az ml compute show ... --query identity.principalId'
```

- Optional fallback: Workspace principal object ID
```powershell
$workspaceObjectId = az ml workspace show `
  --name $WORKSPACE_NAME `
  --resource-group $RESOURCE_GROUP `
  --query identity.principalId -o tsv
$workspaceObjectId
```

- Pick the principal to assign (compute preferred)
```powershell
$principalObjectId = if ($computeObjectId) { $computeObjectId } else { $workspaceObjectId }
$principalObjectId
```

- Optional: Container scope (if you want to scope to one container)
```powershell
$CONTAINER_NAME = "<container-name>"
$containerScope = "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/$STORAGE_ACCOUNT/blobServices/default/containers/$CONTAINER_NAME"
$containerScope
```

Scope reference
- Storage account scope: /subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.Storage/storageAccounts/{acct}
- Container scope: …/blobServices/default/containers/{container}

---

## 2) Assign a Blob data role (RBAC)

Choose the least privilege you need:
- Read-only: Storage Blob Data Reader
- Read/write/delete: Storage Blob Data Contributor

Assign at the storage account scope (recommended)
```powershell
# Reader (read/list)
az role assignment create `
  --role "Storage Blob Data Reader" `
  --assignee-object-id $principalObjectId `
  --assignee-principal-type ServicePrincipal `
  --scope $storageScope

# OR Contributor (read/write/delete)
az role assignment create `
  --role "Storage Blob Data Contributor" `
  --assignee-object-id $principalObjectId `
  --assignee-principal-type ServicePrincipal `
  --scope $storageScope
```

Assign at a specific container (fine-grained)
```powershell
az role assignment create `
  --role "Storage Blob Data Contributor" `
  --assignee-object-id $principalObjectId `
  --assignee-principal-type ServicePrincipal `
  --scope $containerScope
```

Important: --scope must be the full resource ID, not just the storage account name.

---

## 3) Verify the assignment

Allow 2–8 minutes for RBAC propagation, then:

```powershell
# Verify at storage account scope
az role assignment list `
  --assignee $principalObjectId `
  --scope $storageScope `
  -o table

# Or at container scope
# az role assignment list --assignee $principalObjectId --scope $containerScope -o table
```

---

## Troubleshooting and tips
- If `identity.principalId` is empty, enable the identity (compute/workspace) and re-fetch.
- Ensure the right subscription is set: `az account set --subscription $SUBSCRIPTION_ID`.
- Install the ML extension if missing: `az extension add -n ml`.
- Prefer Managed Identity + RBAC; avoid using storage account keys or embedding secrets.
- Use Reader for read-only scenarios; use Contributor when the compute needs to write logs/artifacts or create containers.
- For code running on AML compute, use DefaultAzureCredential/ManagedIdentityCredential to access Blob with the managed identity—no keys needed.
