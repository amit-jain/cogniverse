# Backend Profile Management - Dashboard Guide

This guide shows how to manage backend profiles (video processing configurations) using the Cogniverse dashboard.

## Overview

Backend profiles define how videos are processed and indexed in Cogniverse. Each profile specifies:

- **Schema**: Vespa schema template for document structure
- **Embedding Model**: Model used for generating embeddings (e.g., ColPali, VideoPrism)
- **Embedding Type**: Processing approach (multi_vector, single_vector)
- **Strategies**: Processing strategy configurations (segmentation, embedding, etc.)
- **Pipeline Configuration**: Processing pipeline settings
- **Schema Config**: Schema metadata (embedding dimensions, model name, patch count, etc.)
- **Model-Specific Config**: Optional model-specific parameters (e.g., quantization, batch size)

Profiles are **tenant-scoped**, allowing each tenant to have isolated configurations.

## Accessing Backend Profiles

1. Launch the dashboard:
   ```bash
   uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py --server.port 8501
   ```

2. Set the **Active Tenant** in the sidebar. This is required — the dashboard
   blocks every tab, including Backend Profiles, until a tenant is entered
   here. There is no default tenant fallback.

3. Navigate to the **⚙️ Configuration** tab, then select the **🔧 Backend Profiles** sub-tab (5th sub-tab). The Configuration tab also shows an editable **Tenant ID** text field, pre-filled from the sidebar's Active Tenant, that applies to all its sub-tabs.

## Creating a Profile

### Step 1: Click "Create New Profile"

The create form will appear with the following fields:

### Step 2: Fill Required Fields

**Profile Identity:**

- **Profile Name**: Unique identifier (e.g., `video_colpali_mv_frame`)
  - Use naming convention: `{type}_{model}_{variant}_{strategy}`
  - Must be unique within the tenant

- **Type**: Profile type — must be one of `video`, `image`, `audio`, `document`, `code`

- **Description**: Human-readable description (optional but recommended)

**Schema Configuration:**

- **Schema Name**: Vespa schema template name
  - Must exist in your schema directory
  - Example: `video_colpali_smol500_mv_frame`, `video_videoprism_base_mv_chunk_30s`

- **Schema Config** (optional JSON, defaults to `{}`): Schema metadata such as `embedding_dim`, `model_name`, `num_patches`, `binary_dim`
  - If `embedding_dim` is set, it must be an integer between 1 and 100000

**Embedding Configuration:**

- **Embedding Model**: Model identifier
  - Format: `org/model-name` (e.g., `vidore/colpali`)
  - Or simple name (e.g., `videoprism-base`)

- **Embedding Type**: Processing approach
  - `multi_vector`: Multi-vector embedding (frames or chunks)
  - `single_vector`: Single embedding for entire video

**Strategy Configuration (Optional):**

- **Strategies**: JSON object mapping strategy names to configurations
  ```json
  {
    "segmentation": {
      "class": "FrameSegmentationStrategy",
      "params": {
        "fps": 1.0,
        "threshold": 0.999,
        "max_frames": 3000
      }
    },
    "embedding": {
      "class": "MultiVectorEmbeddingStrategy",
      "params": {
        "model_name": "TomoroAI/tomoro-colqwen3-embed-4b"
      }
    }
  }
  ```

**Pipeline Configuration (Optional):**

- **Pipeline Config**: JSON object for processing settings
  ```json
  {
    "extract_keyframes": true,
    "transcribe_audio": true,
    "generate_descriptions": true,
    "generate_embeddings": true,
    "keyframe_strategy": "fps",
    "keyframe_fps": 1.0
  }
  ```

**Model-Specific Configuration (Optional):**

- **Model Specific**: JSON object for model parameters
  ```json
  {
    "quantization": "int8",
    "max_batch_size": 32
  }
  ```

**Deployment (Optional):**

- **Deploy Schema Immediately**: Checkbox — when checked, the schema is deployed to the backend as part of profile creation (equivalent to setting `deploy_schema: true` in the create API request). Leave unchecked to deploy later from the Deploy Schema tab.

### Step 3: Submit

Click **Create Profile** button. You'll see:

- Success message with profile name

- Profile appears in the dropdown selector

- Automatic validation of all fields

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

1. Select the profile from the dropdown
2. Navigate to the **Edit** tab
3. Modify any of the 4 mutable fields
4. Click **Save Changes**

Every update is versioned — each write creates a new, incrementing version number that is visible via the profile detail response and the History sub-tab. There is no client-supplied version check: updates are not rejected for being based on a stale read, and two concurrent writers will silently overwrite each other (the last write wins). A single dashboard/runtime process serializes its own writes with an internal lock, but this does not protect against concurrent writes from separate processes.

## Deploying a Schema

Deploying a schema creates the Vespa document schema in your configured backend.

### Prerequisites

1. Profile must exist
2. System config must have valid backend URL
3. Schema template must exist in schema directory

### Deploy Steps

1. Select the profile from the dropdown
2. Navigate to the **Deploy Schema** tab
3. Review deployment settings:
   - **Force Redeployment**: Redeploy even if already deployed
4. Click **Deploy Schema**

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

The profile details page automatically displays:

- **Schema Status**: ✅ Deployed / ⚠️ Not Deployed / Unknown

- **Tenant Schema Name**: Full schema name in Vespa (shown in tooltip when deployed)

- **Error messages**: Displayed if API connection fails

The status is refreshed automatically when you view the profile.

## Deleting a Profile

### Delete Options

1. **Delete Profile Only**: Remove from database, keep schema in Vespa
2. **Delete Profile + Schema**: Remove both profile and Vespa schema

### Delete Steps

1. Select the profile from the dropdown
2. Navigate to the **Delete** tab
3. Choose deletion scope:
   - ☐ Also delete associated schema from backend
4. Type the profile name to confirm
5. Click **Delete Profile**

### Safety Features

- Confirmation dialog prevents accidental deletion
- Deletion is **permanent** - no undo
- Schema deletion fails safely if schema doesn't exist
- Tenant isolation prevents cross-tenant deletion

## Viewing Profile Details

Select a profile from the dropdown to view:

**Summary Metrics:**

- Type

- Embedding Type

- Schema Name

- Schema Status (✅ Deployed / ⚠️ Not Deployed)

**Description:** Displayed below metrics if available

**Detailed Configuration:** Access via Edit, Deploy Schema, or Delete tabs to view and modify full profile configuration including strategies and pipeline config.

## Multi-Tenant Isolation

Profiles are **strictly isolated** by tenant:

- Each tenant sees only their own profiles
- Same profile name can exist in different tenants
- Cannot access, edit, or delete other tenants' profiles
- Tenant ID is required on every operation — there is no default/fallback tenant. Omitting it raises an error via the API and blocks every tab in the dashboard

Example:
```text
tenant_a → video_colpali_mv_frame (model: vidore/colpali)
tenant_b → video_colpali_mv_frame (model: custom/model)
```

Both can coexist without conflict.

## Common Workflows

### Workflow 1: Create and Deploy

1. Create profile with all required fields
2. Verify profile appears in dropdown
3. Select profile and go to Deploy Schema tab
4. Click Deploy Schema and wait for confirmation
5. Use profile name in ingestion scripts

### Workflow 2: Test with Different Settings

1. Create profile with base settings
2. Select profile and deploy schema
3. Test ingestion/query
4. Go to Edit tab and modify pipeline_config
5. Go to Deploy Schema tab and enable Force Redeployment
6. Compare results

### Workflow 3: Clone for Different Tenant

1. Fetch the profile JSON from tenant_a via `GET /admin/profiles/{profile_name}?tenant_id=tenant_a` (the dashboard's Import/Export tab exports a tenant's whole configuration, not a single profile)
2. Switch the Active Tenant to tenant_b
3. Create profile with same config
4. Deploy (creates tenant_b-specific schema)
5. Both tenants have isolated instances

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

All dashboard operations can be performed via REST API. See [Profile API Reference](profile-api-reference.md) for details.

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
   - Good: `video_videoprism_base_mv_chunk_30s`
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
