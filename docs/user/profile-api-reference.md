# Backend Profile Management API Reference

REST API documentation for managing backend profiles (video processing configurations).

## Base URL

```text
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

Or for business-rule validation errors raised by `ProfileValidator` (e.g. duplicate profile name, invalid profile name characters, unknown embedding type, missing schema template):

```json
{
  "detail": {
    "message": "Profile validation failed",
    "errors": [
      "Profile 'video_test' already exists for tenant 'acme_corp'",
      "Invalid embedding type 'wrong_type'. Must be one of: ['multi_vector', 'single_vector']"
    ]
  }
}
```

Or for request-schema validation errors (missing/malformed required field, caught by FastAPI/Pydantic before the route body runs):

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "tenant_id"],
      "msg": "Field required",
      "input": {"profile_name": "test", "...": "..."}
    }
  ]
}
```

Common status codes:

- `400`: Bad request (business-rule validation error — e.g. duplicate profile name, invalid profile name, unknown embedding type, missing schema template)

- `404`: Resource not found

- `409`: Conflict — only returned by Delete Profile when `delete_schema=true` and another profile still references the same schema

- `422`: Unprocessable entity (request body missing a required field or failing Pydantic type validation, e.g. `tenant_id` omitted entirely)

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
  "type": "string (optional, default: 'video')",
  "schema_name": "string (required)",
  "embedding_model": "string (required)",
  "embedding_type": "string (required, enum: multi_vector|single_vector)",
  "description": "string (optional, default: '')",
  "strategies": {
    "segmentation": {
      "class": "string",
      "params": {}
    },
    "embedding": {
      "class": "string",
      "params": {}
    }
  },
  "pipeline_config": {
    "frame_extraction": {
      "fps": 1,
      "max_frames": 100
    }
  },
  "model_specific": {
    "quantization": "int8"
  },
  "schema_config": {
    "embedding_dim": 128,
    "num_patches": 1024
  },
  "deploy_schema": "boolean (optional, default: false)"
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
    "schema_name": "video_colpali_smol500_mv_frame",
    "embedding_model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "embedding_type": "multi_vector",
    "description": "ColPali model with frame-based embedding for video search",
    "strategies": {
      "segmentation": {
        "class": "FrameSegmentationStrategy",
        "params": {"fps": 1.0, "max_frames": 100}
      },
      "embedding": {
        "class": "MultiVectorEmbeddingStrategy",
        "params": {}
      }
    },
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
  "schema_deployed": false,
  "tenant_schema_name": null,
  "created_at": "2024-01-15T10:00:00.000Z",
  "version": 1
}
```

**Validation Rules:**

- `profile_name`: Must be unique within tenant, max 100 chars, alphanumeric + underscore + hyphen only
- `tenant_id`: Required, non-empty — identifies the tenant owning the profile
- `type`: Optional (defaults to "video"), must be one of: "video", "image", "audio", "document", "code"
- `schema_name`: Must have a matching template file `{schema_name}_schema.json` in the configured schema templates directory (defaults to `configs/schemas/`); the template must contain top-level `name` and `document.fields`
- `embedding_model`: Format `org/model` or `model-name`
- `embedding_type`: Must be `multi_vector` or `single_vector`
- `strategies`: Optional (defaults to empty dict); each entry must be an object with a `class` key naming an importable strategy class
- `pipeline_config`: Optional (defaults to empty dict), must be valid JSON object
- `model_specific`: Optional (defaults to null), must be valid JSON object
- `schema_config`: Optional (defaults to empty dict); if it includes `embedding_dim`, the value must be an integer between 1 and 100000
- `deploy_schema`: Optional (defaults to false), boolean flag

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
  "profiles": [
    {
      "profile_name": "video_colpali_mv_frame",
      "type": "video",
      "schema_name": "video_colpali_smol500_mv_frame",
      "embedding_model": "TomoroAI/tomoro-colqwen3-embed-4b",
      "description": "ColPali model with frame-based embedding",
      "schema_deployed": true,
      "created_at": "2024-01-15T10:00:00.000Z"
    },
    {
      "profile_name": "video_videoprism_global",
      "type": "video",
      "schema_name": "video_videoprism_base_mv_chunk_30s",
      "embedding_model": "videoprism_public_v1_base_hf",
      "description": "VideoPrism global embeddings",
      "schema_deployed": false,
      "created_at": "2024-01-15T10:00:00.000Z"
    }
  ],
  "total_count": 2,
  "tenant_id": "acme_corp"
}
```

**Notes:**

- Returns empty array if no profiles exist for tenant

- Profiles are tenant-isolated (cannot see other tenants' profiles)

- `created_at` in this list response is NOT a persisted per-profile creation timestamp — the config store doesn't track it, so the route fills it in as the current time of the list request. Every profile in a given response therefore shows the same `created_at`. Use Get Profile for a real per-profile version/timestamp derived from the config store's version history.

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
  "schema_name": "video_colpali_smol500_mv_frame",
  "embedding_model": "TomoroAI/tomoro-colqwen3-embed-4b",
  "embedding_type": "multi_vector",
  "description": "ColPali model with frame-based embedding for video search",
  "strategies": {
    "segmentation": {
      "class": "FrameSegmentationStrategy",
      "params": {"fps": 1.0, "max_frames": 100}
    },
    "embedding": {
      "class": "MultiVectorEmbeddingStrategy",
      "params": {}
    }
  },
  "pipeline_config": {
    "frame_extraction": {
      "fps": 1,
      "max_frames": 100
    }
  },
  "schema_config": {},
  "model_specific": null,
  "schema_deployed": false,
  "tenant_schema_name": null,
  "created_at": "2024-01-15T10:00:00.000Z",
  "version": 1
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
  "strategies": "object (optional)",
  "pipeline_config": "object (optional)",
  "model_specific": "object (optional)"
}
```

**Mutable Fields:**

- `description`

- `strategies` (Dict[str, Any])

- `pipeline_config` (Dict[str, Any])

- `model_specific`

**Immutable Fields** (cannot be updated, create new profile instead):

- `type`

- `schema_name`

- `embedding_model`

- `schema_config`

- `model_loader`

`ProfileUpdateRequest` only declares `tenant_id`, `description`, `strategies`, `pipeline_config`, and `model_specific`, so any immutable field name sent in the request body (e.g. `schema_name`) is silently dropped by Pydantic before the route body runs — it never reaches the update logic. If that leaves no recognized fields to change, the request fails with `400 Bad Request: "No fields to update provided"` (see below) rather than a "cannot update immutable field" error.

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

**Error Response:** `400 Bad Request` (no mutable fields provided, e.g. body only contained an immutable field name)

```json
{
  "detail": "No fields to update provided"
}
```

If `strategies` is one of the provided mutable fields, its value is re-validated with the same strategy-class checks used at creation time; a bad `strategies` block returns `400` with `{"message": "Invalid update fields", "errors": [...]}`.

**Concurrency:**

- Updates are serialized via a single in-process lock on the shared `ConfigManager` instance — it guards backend-profile reads/writes for every tenant in that runtime process, not just the tenant being updated; it does not span multiple runtime processes

- The config store versions every write — each successful update creates a new version and `version` in the response is the resulting version number

---

### 5. Delete Profile

Delete a backend profile and optionally its backend schema.

**Endpoint:** `DELETE /admin/profiles/{profile_name}`

**Path Parameters:**

- `profile_name`: Profile identifier

**Query Parameters:**

- `tenant_id` (required): Tenant identifier
- `delete_schema` (optional, default=false): Also delete backend schema

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
  "schema_deleted": true,
  "deleted_at": "2024-01-15T10:30:00.000Z"
}
```

**Error Response:** `404 Not Found`

```json
{
  "detail": "Profile 'video_colpali_mv_frame' not found for tenant 'acme_corp'"
}
```

**Error Response:** `409 Conflict` (only when `delete_schema=true` and another profile in the same tenant still references the schema)

```json
{
  "detail": "Cannot delete schema 'video_colpali_smol500_mv_frame': other profiles using it: ['video_colpali_other_profile']"
}
```

**Notes:**

- Deletion is permanent - no undo

- If `delete_schema=true` but schema doesn't exist, operation still succeeds

- Cannot delete other tenants' profiles

---

### 6. Deploy Schema

Deploy backend schema for a profile to the configured backend.

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
  "schema_name": "video_colpali_smol500_mv_frame",
  "tenant_schema_name": "video_colpali_smol500_mv_frame_acme_corp_acme_corp",
  "deployment_status": "success",
  "deployed_at": "2024-01-15T10:30:00.000Z"
}
```

**Error Response:** `200 OK` (with failed status)

```json
{
  "profile_name": "video_colpali_mv_frame",
  "tenant_id": "acme_corp",
  "schema_name": "video_colpali_smol500_mv_frame",
  "tenant_schema_name": "",
  "deployment_status": "failed",
  "deployed_at": "2024-01-15T10:30:00.000Z",
  "error_message": "Connection to Vespa refused"
}
```

**Deployment Process:**

1. Check if schema already exists (skip if exists and force=false)
2. Call backend.schema_registry.deploy_schema() with tenant_id, base_schema_name, optional config, and optional force parameter
3. Generate tenant-specific schema name via `backend.get_tenant_schema_name(tenant_id, base_schema_name)`: the tenant_id is first canonicalized to `org:tenant` form (a bare tenant like `acme_corp` canonicalizes to `acme_corp:acme_corp`; an explicit `org:tenant` value like `acme:prod` is used as-is), then the colon is replaced with an underscore and appended — so `tenant_id="acme_corp"` yields `{base_schema_name}_acme_corp_acme_corp`, and `tenant_id="acme:prod"` yields `{base_schema_name}_acme_prod`
4. Return deployment status ("success", "failed", or "already_deployed")

**Prerequisites:**

- Profile must exist

- Schema template must exist in configured schema directory

- Backend must be accessible

- System config must have valid `backend_url`

---

## Request/Response Schemas

### ProfileCreateRequest

```typescript
{
  profile_name: string,        // Required, unique within tenant
  tenant_id: string,           // Required: tenant identifier for isolation
  type?: string,               // Optional (default: "video")
  schema_name: string,         // Required, must exist in schema dir
  embedding_model: string,     // Required (e.g., "TomoroAI/tomoro-colqwen3-embed-4b")
  embedding_type: "multi_vector" | "single_vector",  // Required
  description?: string,        // Optional (default: "")
  strategies?: object,         // Optional (default: {}, Dict[str, Any])
  pipeline_config?: object,    // Optional (default: {}, Dict[str, Any])
  model_specific?: object,     // Optional (default: null, Dict[str, Any])
  schema_config?: object,      // Optional (default: {}, Dict[str, Any])
  deploy_schema?: boolean      // Optional (default: false)
}
```

### ProfileSummary

```typescript
{
  profile_name: string,
  type: string,
  description: string,
  schema_name: string,
  embedding_model: string,
  schema_deployed: boolean,
  created_at: string               // ISO 8601 timestamp
}
```

### ProfileListResponse

```typescript
{
  profiles: ProfileSummary[],      // List of profile summaries
  total_count: number,
  tenant_id: string
}
```

### ProfileDetail

```typescript
{
  profile_name: string,
  tenant_id: string,
  type: string,
  schema_name: string,
  embedding_model: string,
  embedding_type: string,
  description: string,
  strategies: object,              // Dict[str, Any]
  pipeline_config: object,         // Dict[str, Any]
  schema_config: object,           // Dict[str, Any]
  model_specific: object | null,
  schema_deployed: boolean,
  tenant_schema_name: string | null,
  created_at: string,              // ISO 8601 timestamp
  version: number
}
```

### ProfileUpdateRequest

```typescript
{
  tenant_id: string,           // Required
  description?: string,        // Optional
  strategies?: object,         // Optional (Dict[str, Any])
  pipeline_config?: object,    // Optional (Dict[str, Any])
  model_specific?: object      // Optional (Dict[str, Any])
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
  schema_deleted: boolean,
  deleted_at: string             // ISO 8601 timestamp
}
```

### SchemaDeploymentRequest

```typescript
{
  tenant_id: string,  // Required
  force: boolean      // Optional, default=false
}
```

### SchemaDeploymentResponse

```typescript
{
  profile_name: string,
  tenant_id: string,
  schema_name: string,
  tenant_schema_name: string,
  deployment_status: string,      // "success" | "failed" | "already_deployed"
  deployed_at: string,            // ISO 8601 timestamp
  error_message?: string          // Only present if deployment_status is "failed"
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
    "schema_name": "video_colpali_smol500_mv_frame",
    "embedding_model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "embedding_type": "multi_vector",
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
import requests

class ProfileClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def create_profile(self, profile: dict) -> dict:
        response = self.session.post(
            f"{self.base_url}/admin/profiles",
            json=profile,
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()

    def list_profiles(self, tenant_id: str) -> list:
        response = self.session.get(
            f"{self.base_url}/admin/profiles",
            params={"tenant_id": tenant_id},
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()["profiles"]

    def deploy_schema(self, profile_name: str, tenant_id: str, force: bool = False) -> dict:
        response = self.session.post(
            f"{self.base_url}/admin/profiles/{profile_name}/deploy",
            json={"tenant_id": tenant_id, "force": force},
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()

# Usage
client = ProfileClient()
profile = client.create_profile({
    "profile_name": "video_test",
    "tenant_id": "my_tenant",
    "type": "video",
    "schema_name": "video_colpali_smol500_mv_frame",
    "embedding_model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "embedding_type": "multi_vector"
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
  schema_name: "video_colpali_smol500_mv_frame",
  embedding_model: "TomoroAI/tomoro-colqwen3-embed-4b",
  embedding_type: "multi_vector"
});
```

---

## Next Steps

- [Profile Management Dashboard](profile-management.md) - UI guide
- [Dynamic Profiles Architecture](../architecture/dynamic-profiles.md) - System design
