# Backend Profile Management - Web Client Guide

This guide shows how to manage backend profiles (video processing configurations) in the web client's **Backend profiles** view.

## Overview

Backend profiles define how videos are processed and indexed in Cogniverse. Each profile specifies:

- **Schema**: Vespa schema template for document structure
- **Embedding Model**: Model used for generating embeddings (e.g., ColPali, X-CLIP)
- **Embedding Type**: Processing approach (multi_vector, single_vector)
- **Strategies**: Processing strategy configurations (segmentation, embedding, etc.)
- **Pipeline Configuration**: Processing pipeline settings
- **Schema Config**: Schema metadata (embedding dimensions, model name, patch count, etc.)
- **Model-Specific Config**: Optional model-specific parameters (e.g., quantization, batch size)

Profiles are **tenant-scoped**, allowing each tenant to have isolated configurations.

## Accessing Backend Profiles

1. Open the web client (http://localhost:28400 under `cogniverse up`; see
   [Web Client](../modules/web-client.md)).
2. Choose **Backend profiles** under Operations.
3. Enter the **Tenant ID** (known tenants are suggested) and click
   **Show profiles**. There is no default tenant.

The **Profiles of {tenant}** panel lists the profiles created for the tenant
with their type, schema, embedding model, whether the schema is deployed, and
description. Shipped profiles are not listed. **Refresh** reloads the list.

## Creating a Profile

The **New profile for {tenant}** panel creates one.

### Step 1: Choose a Starting Point

**Start from shipped profile** lists the shipped profiles. Choosing one fills
every field from it; **Blank profile** clears them.

### Step 2: Fill the Fields

| Field | Required | Meaning |
|---|---|---|
| Profile name | yes | Unique within the tenant |
| Type | yes | `video`, `image`, `audio`, `document` or `code` |
| Schema name | yes | Base schema template, e.g. `video_colpali_smol500_mv_frame` |
| Embedding model | yes | e.g. `TomoroAI/tomoro-colqwen3-embed-4b` |
| Embedding type | yes | `multi_vector` or `single_vector` |
| Model loader | no | Suggestions come from the shipped profiles |
| Process type | no | Empty lets the runtime infer it |
| Description | no | |
| Pipeline config, Strategies, Schema config, Model-specific parameters, Extra config | no | JSON objects; empty means `{}` (model-specific: none) |

Tick **Deploy the schema now** to deploy the tenant schema in the same request.

### Step 3: Submit

Click **Create profile**. The view reports the created profile and its config
version (and the deployed tenant schema when requested), lists it, and opens
it. A rejected request shows the runtime's reason in the form.

### Validation Rules

The system validates:

- Profile name is unique within tenant (checked on create only)

- Profile name contains only alphanumeric characters, underscores, and hyphens (max 100 chars)

- Profile type is one of: `video`, `image`, `audio`, `document`, `code`

- Schema name exists in schema directory (`configs/schemas/{schema_name}_schema.json`)

- Embedding model is a non-empty string (a warning, not a hard error, is logged if it doesn't look like `org/model` or `model-name`)

- Embedding type is a valid enum value (`multi_vector` or `single_vector`)

- Each strategy's `class` is importable (e.g., `FrameSegmentationStrategy`)

- If `schema_config.embedding_dim` is set, it must be an integer between 1 and 100000

- JSON fields are valid JSON

- Required fields are not empty

## Editing a Profile

### Mutable Fields

Only these fields can be updated after creation:

- Description

- Strategies

- Pipeline Configuration

- Model-Specific Configuration

**Immutable fields** (require creating a new profile):

- Profile Name (cannot be changed - path parameter)

- Type

- Schema Name

- Embedding Model

- Schema Config

### Edit Steps

1. Click the profile's name in the list. Its panel shows type, schema, the
   tenant schema it is deployed as, embedding model and type, model loader,
   process type, config version, schema config and extra config.
2. In **Edit**, change the description or the Pipeline config, Strategies or
   Model-specific parameters JSON. Model-specific parameters cannot be removed;
   enter `{}` to clear them.
3. Click **Save changes**. Only changed fields are sent; the view reports the
   saved fields and the new config version, or "Nothing to save; no field
   changed."

Every update is versioned — each write creates a new, incrementing version number that is shown as the profile panel's **Config version**. There is no client-supplied version check: updates are not rejected for being based on a stale read, and two concurrent writers will silently overwrite each other (the last write wins). A single runtime process serializes its own writes with an internal lock, but this does not protect against concurrent writes from separate processes.

## Deploying a Schema

Deploying a schema creates the Vespa document schema in your configured backend.

### Prerequisites

1. Profile must exist
2. System config must have valid backend URL
3. Schema template must exist in schema directory

### Deploy Steps

1. Open the profile from the list
2. Under **Schema**, tick **Redeploy even if already deployed** to force a redeployment
3. Click **Deploy schema**

The view reports the tenant schema it deployed, that it was already deployed,
or the runtime's error message when deployment failed.

### Deployment Process

The system will:
1. Generate a tenant-specific schema name. The tenant ID is canonicalized to
   `org:tenant` form first (a simple ID like `acme` becomes `acme:acme`), then
   the colon is replaced with an underscore and appended to the base schema
   name — e.g., schema `video_colpali` + tenant `acme` → `video_colpali_acme_acme`;
   tenant `acme:prod` → `video_colpali_acme_prod`
2. Skip deployment and report `already_deployed` if the schema already exists and Force Redeployment is off
3. Load schema template from disk and apply profile-specific configurations
4. Submit to Vespa via the schema registry
5. Return the deployment status (`success`, `failed`, or `already_deployed`)

### Deployment Status

The list's **Schema deployed** column and the profile panel's **Deployed as**
field (the tenant schema name, or "not deployed") show the current state each
time the profile is loaded.

## Deleting a Profile

### Delete Options

1. **Delete Profile Only**: Remove from database, keep schema in Vespa
2. **Delete Profile + Schema**: Remove both profile and Vespa schema

### Delete Steps

1. Open the profile from the list
2. Under **Delete**, tick **Also delete schema {schema}** to remove the schema too
3. Click **Delete**, type the profile name, and confirm

The view reports what was deleted, including when the schema was not deployed.
Deletion is permanent.

## Multi-Tenant Isolation

Profiles are **strictly isolated** by tenant:

- Each tenant sees only their own profiles
- Same profile name can exist in different tenants
- Cannot access, edit, or delete other tenants' profiles
- Tenant ID is required on every operation — there is no default/fallback tenant. Omitting it raises an error via the API, and the view shows nothing until a tenant is chosen

Example:
```text
tenant_a → video_colpali_mv_frame (model: vidore/colpali)
tenant_b → video_colpali_mv_frame (model: custom/model)
```

Both can coexist without conflict.

## Common Workflows

### Workflow 1: Create and Deploy

1. Choose the tenant, pick a shipped profile to start from, and set a new profile name
2. Tick **Deploy the schema now** and click **Create profile**
3. Use the profile name in ingestion (the Ingestion view takes a profile name)

### Workflow 2: Test with Different Settings

1. Create the profile and deploy its schema
2. Test ingestion and queries
3. Edit its pipeline config and **Save changes**
4. Tick **Redeploy even if already deployed** and **Deploy schema**
5. Compare results

### Workflow 3: Clone for Different Tenant

1. Fetch the profile JSON from tenant_a via `GET /admin/profiles/{profile_name}?tenant_id=tenant_a`
2. Choose tenant_b in the view
3. Create a profile with the same fields
4. Deploy it (creates the tenant_b schema)

## Troubleshooting

### "Profile already exists"
- Profile name must be unique within tenant
- Choose a different name or delete existing profile

### "Schema not found"
- Ensure schema template exists in configured schema directory
- Check schema name matches file: `{schema_name}_schema.json`

### "Deployment failed"
- Check Vespa backend is running
- Verify backend URL in System Config
- Check schema template is valid JSON
- Review error message for details

### "Cannot update profile"
- Trying to update immutable field (use create instead)
- Profile doesn't exist (check tenant ID)
- Validation error (check JSON syntax)

### Changes appear lost after a concurrent edit
- Updates are not conflict-checked — the last write wins, and the profile's version number simply keeps incrementing
- If two people (or two browser tabs) edit the same profile at once, reload the profile before editing again and re-apply your change

## API Alternative

Every operation in the view is a runtime REST call. See [Profile API Reference](profile-api-reference.md) for details.

Example:
```bash
# Create profile via API
curl -X POST http://localhost:8000/admin/profiles \
  -H "Content-Type: application/json" \
  -d '{
    "profile_name": "video_colpali_custom",
    "tenant_id": "my_tenant",
    "type": "video",
    "schema_name": "video_colpali_smol500_mv_frame",
    "embedding_model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "embedding_type": "multi_vector"
  }'
```

## Best Practices

1. **Naming Convention**: Use descriptive, structured names
   - Good: `video_xclip_sv_chunk_6s`
   - Bad: `my_profile_v2`

2. **Documentation**: Always add meaningful descriptions
   - Explain the use case and expected performance

3. **Testing**: Test profiles with sample videos before production
   - Use `--max-frames 1` for quick validation

4. **Version Control**: Export profile configurations for tracking
   ```bash
   # List all profiles for a tenant
   curl http://localhost:8000/admin/profiles?tenant_id=my_tenant

   # Get detailed profile configuration
   curl http://localhost:8000/admin/profiles/my_profile?tenant_id=my_tenant > profile.json
   ```

5. **Schema Organization**: Keep schema templates in version control
   - Schema directory: `configs/schemas/`
   - Use git to track schema changes

6. **Tenant Strategy**: Use meaningful tenant IDs
   - Good: `customer_acme`, `team_research`
   - Bad: `tenant1`, `test`
   - Tenant IDs may be a simple name (`acme`) or `org:tenant` form (`acme:production`); both are canonicalized to `org:tenant` internally

## Next Steps

- [Profile API Reference](profile-api-reference.md) - REST API documentation
- [Dynamic Profiles Architecture](../architecture/dynamic-profiles.md) - System design
