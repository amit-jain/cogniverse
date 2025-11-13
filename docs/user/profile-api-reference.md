# Backend Profile Management API Reference

REST API documentation for managing backend profiles (video processing configurations).

## Base URL

```
http://localhost:8000/admin
```

All endpoints are under the `/admin` prefix.

## Authentication

Currently no authentication required. Future versions will require API keys or OAuth tokens.

## Common Headers

```http
Content-Type: application/json
Accept: application/json
```

## Error Responses

All endpoints return standard HTTP error responses:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Or for validation errors:

```json
{
  "detail": {
    "message": "Invalid request",
    "errors": [
      "Field 'profile_name' is required",
      "Field 'embedding_type' must be one of: frame_based, chunk_based, global"
    ]
  }
}
```

Common status codes:
- `400`: Bad request (validation error)
- `404`: Resource not found
- `409`: Conflict (duplicate profile name)
- `500`: Internal server error

---

## Endpoints

### 1. Create Profile

Create a new backend profile for a tenant.

**Endpoint:** `POST /admin/profiles`

**Request Body:**

```json
{
  "profile_name": "string (required)",
  "tenant_id": "string (required)",
  "type": "string (required)",
  "schema_name": "string (required)",
  "embedding_model": "string (required)",
  "embedding_type": "string (required, enum: frame_based|chunk_based|global)",
  "description": "string (optional)",
  "strategies": [
    {
      "name": "string",
      "type": "string",
      "ranking": "string"
    }
  ],
  "pipeline_config": {
    "frame_extraction": {
      "fps": 1,
      "max_frames": 100
    }
  },
  "model_specific": {
    "quantization": "int8"
  }
}
```

**Example Request:**

```bash
curl -X POST http://localhost:8000/admin/profiles \
  -H "Content-Type: application/json" \
  -d '{
    "profile_name": "video_colpali_mv_frame",
    "tenant_id": "acme_corp",
    "type": "video",
    "schema_name": "video_colpali_base",
    "embedding_model": "vidore/colpali",
    "embedding_type": "frame_based",
    "description": "ColPali model with frame-based embedding for video search",
    "strategies": [
      {
        "name": "multimodal_fusion",
        "type": "multimodal",
        "ranking": "bm25_semantic_fusion"
      }
    ],
    "pipeline_config": {
      "frame_extraction": {
        "fps": 1,
        "max_frames": 100
      }
    }
  }'
```

**Response:** `201 Created`

```json
{
  "profile_name": "video_colpali_mv_frame",
  "tenant_id": "acme_corp",
  "type": "video",
  "schema_name": "video_colpali_base",
  "embedding_model": "vidore/colpali",
  "embedding_type": "frame_based",
  "description": "ColPali model with frame-based embedding for video search",
  "strategies": [
    {
      "name": "multimodal_fusion",
      "type": "multimodal",
      "ranking": "bm25_semantic_fusion"
    }
  ],
  "pipeline_config": {
    "frame_extraction": {
      "fps": 1,
      "max_frames": 100
    }
  },
  "model_specific": null,
  "version": 1
}
```

**Validation Rules:**

- `profile_name`: Must be unique within tenant, alphanumeric + underscore
- `tenant_id`: Required, non-empty
- `type`: Required, typically "video", "image", or "audio"
- `schema_name`: Must exist in schema directory
- `embedding_model`: Format `org/model` or `model-name`
- `embedding_type`: Must be `frame_based`, `chunk_based`, or `global`
- `strategies`: Optional, must be valid JSON array
- `pipeline_config`: Optional, must be valid JSON object
- `model_specific`: Optional, must be valid JSON object

---

### 2. List Profiles

Get all backend profiles for a tenant.

**Endpoint:** `GET /admin/profiles`

**Query Parameters:**

- `tenant_id` (required): Tenant identifier

**Example Request:**

```bash
curl "http://localhost:8000/admin/profiles?tenant_id=acme_corp"
```

**Response:** `200 OK`

```json
{
  "tenant_id": "acme_corp",
  "profiles": [
    {
      "profile_name": "video_colpali_mv_frame",
      "tenant_id": "acme_corp",
      "type": "video",
      "schema_name": "video_colpali_base",
      "embedding_model": "vidore/colpali",
      "embedding_type": "frame_based",
      "description": "ColPali model with frame-based embedding",
      "strategies": [...],
      "pipeline_config": {...},
      "model_specific": null,
      "version": 1
    },
    {
      "profile_name": "video_videoprism_global",
      "tenant_id": "acme_corp",
      "type": "video",
      "schema_name": "video_videoprism_base",
      "embedding_model": "videoprism-base",
      "embedding_type": "global",
      "description": "VideoPrism global embeddings",
      "strategies": [...],
      "pipeline_config": {...},
      "model_specific": null,
      "version": 1
    }
  ]
}
```

**Notes:**
- Returns empty array if no profiles exist for tenant
- Profiles are tenant-isolated (cannot see other tenants' profiles)

---

### 3. Get Profile

Get a specific backend profile by name.

**Endpoint:** `GET /admin/profiles/{profile_name}`

**Path Parameters:**

- `profile_name`: Profile identifier

**Query Parameters:**

- `tenant_id` (required): Tenant identifier

**Example Request:**

```bash
curl "http://localhost:8000/admin/profiles/video_colpali_mv_frame?tenant_id=acme_corp"
```

**Response:** `200 OK`

```json
{
  "profile_name": "video_colpali_mv_frame",
  "tenant_id": "acme_corp",
  "type": "video",
  "schema_name": "video_colpali_base",
  "embedding_model": "vidore/colpali",
  "embedding_type": "frame_based",
  "description": "ColPali model with frame-based embedding for video search",
  "strategies": [
    {
      "name": "multimodal_fusion",
      "type": "multimodal",
      "ranking": "bm25_semantic_fusion"
    }
  ],
  "pipeline_config": {
    "frame_extraction": {
      "fps": 1,
      "max_frames": 100
    }
  },
  "model_specific": null,
  "version": 1,
  "schema_deployed": false,
  "tenant_schema_name": null
}
```

**Error Response:** `404 Not Found`

```json
{
  "detail": "Profile 'video_colpali_mv_frame' not found for tenant 'acme_corp'"
}
```

---

### 4. Update Profile

Update mutable fields of an existing profile.

**Endpoint:** `PUT /admin/profiles/{profile_name}`

**Path Parameters:**

- `profile_name`: Profile identifier

**Request Body:**

```json
{
  "tenant_id": "string (required)",
  "description": "string (optional)",
  "strategies": [...] (optional),
  "pipeline_config": {...} (optional),
  "model_specific": {...} (optional)
}
```

**Mutable Fields:**
- `description`
- `strategies`
- `pipeline_config`
- `model_specific`

**Immutable Fields** (cannot be updated, create new profile instead):
- `profile_name`
- `type`
- `schema_name`
- `embedding_model`
- `embedding_type`

**Example Request:**

```bash
curl -X PUT http://localhost:8000/admin/profiles/video_colpali_mv_frame \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "acme_corp",
    "description": "Updated: ColPali with optimized frame extraction",
    "pipeline_config": {
      "frame_extraction": {
        "fps": 0.5,
        "max_frames": 50
      }
    }
  }'
```

**Response:** `200 OK`

```json
{
  "profile_name": "video_colpali_mv_frame",
  "tenant_id": "acme_corp",
  "updated_fields": ["description", "pipeline_config"],
  "version": 2
}
```

**Error Response:** `400 Bad Request` (trying to update immutable field)

```json
{
  "detail": {
    "message": "Invalid update fields",
    "errors": [
      "Cannot update immutable field: embedding_model"
    ]
  }
}
```

**Concurrency:**
- Uses optimistic concurrency control
- Version number increments on each update
- Concurrent updates are serialized via database locks

---

### 5. Delete Profile

Delete a backend profile and optionally its Vespa schema.

**Endpoint:** `DELETE /admin/profiles/{profile_name}`

**Path Parameters:**

- `profile_name`: Profile identifier

**Query Parameters:**

- `tenant_id` (required): Tenant identifier
- `delete_schema` (optional, default=false): Also delete Vespa schema

**Example Request (profile only):**

```bash
curl -X DELETE "http://localhost:8000/admin/profiles/video_colpali_mv_frame?tenant_id=acme_corp"
```

**Example Request (profile + schema):**

```bash
curl -X DELETE "http://localhost:8000/admin/profiles/video_colpali_mv_frame?tenant_id=acme_corp&delete_schema=true"
```

**Response:** `200 OK`

```json
{
  "profile_name": "video_colpali_mv_frame",
  "tenant_id": "acme_corp",
  "deleted": true,
  "schema_deleted": true,
  "tenant_schema_name": "acme_corp_video_colpali_mv_frame"
}
```

**Error Response:** `404 Not Found`

```json
{
  "detail": "Profile 'video_colpali_mv_frame' not found for tenant 'acme_corp'"
}
```

**Notes:**
- Deletion is permanent - no undo
- If `delete_schema=true` but schema doesn't exist, operation still succeeds
- Cannot delete other tenants' profiles

---

### 6. Deploy Schema

Deploy Vespa schema for a profile to the configured backend.

**Endpoint:** `POST /admin/profiles/{profile_name}/deploy`

**Path Parameters:**

- `profile_name`: Profile identifier

**Request Body:**

```json
{
  "tenant_id": "string (required)",
  "force": "boolean (optional, default=false)"
}
```

**Parameters:**
- `force`: If true, redeploy even if already deployed

**Example Request:**

```bash
curl -X POST http://localhost:8000/admin/profiles/video_colpali_mv_frame/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "acme_corp",
    "force": false
  }'
```

**Response:** `200 OK`

```json
{
  "profile_name": "video_colpali_mv_frame",
  "tenant_id": "acme_corp",
  "success": true,
  "tenant_schema_name": "acme_corp_video_colpali_mv_frame",
  "message": "Schema deployed successfully"
}
```

**Error Response:** `500 Internal Server Error`

```json
{
  "detail": "Schema deployment failed: Connection to Vespa refused"
}
```

**Deployment Process:**

1. Generate tenant-specific schema name: `{tenant_id}_{profile_name}`
2. Load schema template from `{schema_name}_schema.json`
3. Apply profile-specific configurations
4. Submit to Vespa backend via deployment API
5. Wait for deployment confirmation (timeout: 30s)

**Prerequisites:**
- Profile must exist
- Schema template must exist in configured schema directory
- Vespa backend must be accessible
- System config must have valid `backend_url`

---

## Request/Response Schemas

### ProfileCreateRequest

```typescript
{
  profile_name: string,        // Required, unique within tenant
  tenant_id: string,           // Required
  type: string,                // Required (e.g., "video", "image")
  schema_name: string,         // Required, must exist in schema dir
  embedding_model: string,     // Required (e.g., "vidore/colpali")
  embedding_type: "frame_based" | "chunk_based" | "global",  // Required
  description?: string,        // Optional
  strategies?: Strategy[],     // Optional
  pipeline_config?: object,    // Optional
  model_specific?: object      // Optional
}
```

### ProfileResponse

```typescript
{
  profile_name: string,
  tenant_id: string,
  type: string,
  schema_name: string,
  embedding_model: string,
  embedding_type: string,
  description: string | null,
  strategies: Strategy[] | null,
  pipeline_config: object | null,
  model_specific: object | null,
  version: number,
  schema_deployed?: boolean,        // Only in GET single profile
  tenant_schema_name?: string | null // Only in GET single profile
}
```

### ProfileUpdateRequest

```typescript
{
  tenant_id: string,           // Required
  description?: string,        // Optional
  strategies?: Strategy[],     // Optional
  pipeline_config?: object,    // Optional
  model_specific?: object      // Optional
}
```

### ProfileUpdateResponse

```typescript
{
  profile_name: string,
  tenant_id: string,
  updated_fields: string[],    // Fields that were changed
  version: number              // Incremented version
}
```

### ProfileDeleteResponse

```typescript
{
  profile_name: string,
  tenant_id: string,
  deleted: boolean,
  schema_deleted: boolean,
  tenant_schema_name: string | null
}
```

### ProfileDeployRequest

```typescript
{
  tenant_id: string,  // Required
  force: boolean      // Optional, default=false
}
```

### ProfileDeployResponse

```typescript
{
  profile_name: string,
  tenant_id: string,
  success: boolean,
  tenant_schema_name: string,
  message: string
}
```

### Strategy

```typescript
{
  name: string,       // Strategy identifier
  type: string,       // Strategy type
  ranking: string     // Ranking method
}
```

---

## Complete Workflow Example

### 1. Create a new profile

```bash
curl -X POST http://localhost:8000/admin/profiles \
  -H "Content-Type: application/json" \
  -d '{
    "profile_name": "video_test_profile",
    "tenant_id": "test_tenant",
    "type": "video",
    "schema_name": "video_test",
    "embedding_model": "vidore/colpali",
    "embedding_type": "frame_based",
    "description": "Test profile for development"
  }'
```

### 2. Deploy the schema

```bash
curl -X POST http://localhost:8000/admin/profiles/video_test_profile/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "test_tenant",
    "force": false
  }'
```

### 3. Check deployment status

```bash
curl "http://localhost:8000/admin/profiles/video_test_profile?tenant_id=test_tenant"
```

### 4. Update pipeline configuration

```bash
curl -X PUT http://localhost:8000/admin/profiles/video_test_profile \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "test_tenant",
    "pipeline_config": {
      "frame_extraction": {
        "fps": 2,
        "max_frames": 200
      }
    }
  }'
```

### 5. List all profiles for tenant

```bash
curl "http://localhost:8000/admin/profiles?tenant_id=test_tenant"
```

### 6. Delete profile and schema

```bash
curl -X DELETE "http://localhost:8000/admin/profiles/video_test_profile?tenant_id=test_tenant&delete_schema=true"
```

---

## Rate Limiting

Currently no rate limiting. Future versions will implement:
- Per-tenant request limits
- Burst protection
- Deployment throttling

---

## Versioning

API version: `v1` (implicit, no version prefix required)

Breaking changes will be introduced in new API versions (`/v2/admin/profiles`).

---

## Client Libraries

### Python

```python
import httpx

class ProfileClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)

    def create_profile(self, profile: dict) -> dict:
        response = self.client.post(
            f"{self.base_url}/admin/profiles",
            json=profile
        )
        response.raise_for_status()
        return response.json()

    def list_profiles(self, tenant_id: str) -> list:
        response = self.client.get(
            f"{self.base_url}/admin/profiles",
            params={"tenant_id": tenant_id}
        )
        response.raise_for_status()
        return response.json()["profiles"]

    def deploy_schema(self, profile_name: str, tenant_id: str, force: bool = False) -> dict:
        response = self.client.post(
            f"{self.base_url}/admin/profiles/{profile_name}/deploy",
            json={"tenant_id": tenant_id, "force": force}
        )
        response.raise_for_status()
        return response.json()

# Usage
client = ProfileClient()
profile = client.create_profile({
    "profile_name": "video_test",
    "tenant_id": "my_tenant",
    "type": "video",
    "schema_name": "video_test",
    "embedding_model": "vidore/colpali",
    "embedding_type": "frame_based"
})
```

### JavaScript

```javascript
class ProfileClient {
  constructor(baseUrl = "http://localhost:8000") {
    this.baseUrl = baseUrl;
  }

  async createProfile(profile) {
    const response = await fetch(`${this.baseUrl}/admin/profiles`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(profile)
    });
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  }

  async listProfiles(tenantId) {
    const response = await fetch(
      `${this.baseUrl}/admin/profiles?tenant_id=${tenantId}`
    );
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    return data.profiles;
  }

  async deploySchema(profileName, tenantId, force = false) {
    const response = await fetch(
      `${this.baseUrl}/admin/profiles/${profileName}/deploy`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tenant_id: tenantId, force })
      }
    );
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  }
}

// Usage
const client = new ProfileClient();
const profile = await client.createProfile({
  profile_name: "video_test",
  tenant_id: "my_tenant",
  type: "video",
  schema_name: "video_test",
  embedding_model: "vidore/colpali",
  embedding_type: "frame_based"
});
```

---

## Next Steps

- [Profile Management Dashboard](profile-management.md) - UI guide
- [Dynamic Profiles Architecture](../architecture/dynamic-profiles.md) - System design
