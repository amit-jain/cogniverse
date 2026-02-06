# Backend Profile Management - Dashboard Guide

This guide shows how to manage backend profiles (video processing configurations) using the Cogniverse dashboard.

## Overview

Backend profiles define how videos are processed and indexed in Cogniverse. Each profile specifies:

- **Schema**: Vespa schema template for document structure
- **Embedding Model**: Model used for generating embeddings (e.g., ColPali, VideoPrism)
- **Embedding Type**: Processing approach (frame_based, video_chunks, direct_video_segment, single_vector)
- **Strategies**: Processing strategy configurations (segmentation, embedding, etc.)
- **Pipeline Configuration**: Processing pipeline settings

Profiles are **tenant-scoped**, allowing each tenant to have isolated configurations.

## Accessing Backend Profiles

1. Launch the dashboard:
   ```bash
   uv run streamlit run scripts/config_management_tab.py --server.port 8501
   ```

2. Navigate to the **Backend Profiles** tab (5th tab)

3. Select your **Tenant ID** from the dropdown (or enter a new one)

## Creating a Profile

### Step 1: Click "Create New Profile"

The create form will appear with the following fields:

### Step 2: Fill Required Fields

**Profile Identity:**

- **Profile Name**: Unique identifier (e.g., `video_colpali_mv_frame`)
  - Use naming convention: `{type}_{model}_{variant}_{strategy}`
  - Must be unique within the tenant

- **Type**: Profile type (e.g., `video`, `image`, `audio`)

- **Description**: Human-readable description (optional but recommended)

**Schema Configuration:**

- **Schema Name**: Vespa schema template name
  - Must exist in your schema directory
  - Example: `video_colpali_smol500_mv_frame`, `video_videoprism_base_mv_chunk_30s`

**Embedding Configuration:**

- **Embedding Model**: Model identifier
  - Format: `org/model-name` (e.g., `vidore/colpali`)
  - Or simple name (e.g., `videoprism-base`)

- **Embedding Type**: Processing approach
  - `frame_based`: Extract frames, embed individually
  - `video_chunks`: Split video into chunks, embed each
  - `direct_video_segment`: Direct video segment embedding
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
        "model_name": "vidore/colsmol-500m"
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

### Step 3: Submit

Click **Create Profile** button. You'll see:

- Success message with profile name

- Profile appears in the dropdown selector

- Automatic validation of all fields

### Validation Rules

The system validates:

- Profile name is unique within tenant

- Schema name exists in schema directory

- Embedding model format is correct

- Embedding type is valid enum value

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

The system uses **optimistic concurrency control** - each update increments the version number to detect conflicts.

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
1. Generate tenant-specific schema name: `{schema_name}_{tenant_id}`
2. Load schema template from disk
3. Apply profile-specific configurations
4. Submit to Vespa via admin API
5. Wait for deployment confirmation

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
- Tenant ID defaults to "default" if not specified

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

1. Export profile JSON from tenant_a
2. Switch to tenant_b
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

### "Version conflict"
- Another user updated the profile concurrently
- Refresh the page and retry your changes

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
    "embedding_model": "vidore/colsmol-500m",
    "embedding_type": "frame_based",
    "pipeline_config": {},
    "strategies": {},
    "schema_config": {}
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

## Next Steps

- [Profile API Reference](profile-api-reference.md) - REST API documentation
- [Dynamic Profiles Architecture](../architecture/dynamic-profiles.md) - System design
