# Dynamic Backend Profiles - Architecture

Architecture overview of the dynamic profile registration system for backend configurations.

## Overview

The dynamic profile system replaces static `config.json` files with a database-backed, multi-tenant configuration management system. This enables:

- **Runtime profile creation** without code deployment
- **Multi-tenant isolation** for SaaS deployments
- **Concurrent access** with proper locking
- **Version control** for configuration changes
- **Schema deployment automation** via API

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                            │
├─────────────────────┬───────────────────────────────────────────┤
│  Dashboard UI       │  REST API Clients                         │
│  (Streamlit)        │  (Python, JavaScript, curl)               │
└──────────┬──────────┴──────────────┬────────────────────────────┘
           │                         │
           │ HTTP                    │ HTTP
           │                         │
┌──────────▼─────────────────────────▼────────────────────────────┐
│                    FastAPI Runtime Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  Admin Router (/admin/profiles)                                 │
│  ├─ create_profile()        ┌─────────────────────┐            │
│  ├─ list_profiles()         │ ProfileValidator    │            │
│  ├─ get_profile()           │ ├─ Validate schema  │            │
│  ├─ update_profile()        │ ├─ Check fields     │            │
│  ├─ delete_profile()        │ └─ Verify format    │            │
│  └─ deploy_schema()         └─────────────────────┘            │
└──────────┬──────────────────────────────────────────────────────┘
           │
           │ Function calls
           │
┌──────────▼──────────────────────────────────────────────────────┐
│                   Configuration Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  ConfigManager                                                  │
│  ├─ get_backend_config()      ┌────────────────┐               │
│  ├─ set_backend_config()      │ Threading Lock │               │
│  ├─ add_backend_profile() ────┤ (Concurrency)  │               │
│  ├─ update_backend_profile() ─┤                │               │
│  ├─ delete_backend_profile() ─┘                │               │
│  └─ list_backend_profiles()                    │               │
│                                                 │               │
│  BackendConfig                                  │               │
│  ├─ profiles: Dict[str, BackendProfileConfig]  │               │
│  ├─ add_profile()                               │               │
│  ├─ merge_profile()                             │               │
│  └─ to_dict() / from_dict()                     │               │
└──────────┬──────────────────────────────────────────────────────┘
           │
           │ Store interface
           │
┌──────────▼──────────────────────────────────────────────────────┐
│                    Persistence Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  SQLiteConfigStore                                              │
│  ├─ get_config()               ┌────────────────────┐          │
│  ├─ set_config()               │ SQLite Features:   │          │
│  ├─ delete_config()            │ ├─ WAL mode        │          │
│  ├─ list_configs()             │ ├─ BEGIN IMMEDIATE │          │
│  └─ get_config_history()       │ ├─ 30s timeout     │          │
│                                 │ └─ Auto-rollback   │          │
│  Database Schema:               └────────────────────┘          │
│  ┌────────────────────────────────────────────────────┐        │
│  │ configurations                                      │        │
│  ├────────────────────────────────────────────────────┤        │
│  │ id           INTEGER PRIMARY KEY                   │        │
│  │ tenant_id    TEXT NOT NULL                         │        │
│  │ scope        TEXT NOT NULL (backend, agent, etc.)  │        │
│  │ service      TEXT NOT NULL                         │        │
│  │ config_key   TEXT NOT NULL                         │        │
│  │ config_value TEXT NOT NULL (JSON)                  │        │
│  │ version      INTEGER NOT NULL                      │        │
│  │ created_at   TEXT NOT NULL                         │        │
│  │ updated_at   TEXT NOT NULL                         │        │
│  │                                                     │        │
│  │ UNIQUE(tenant_id, scope, service, config_key, ver) │        │
│  │ INDEX(tenant_id, scope)                            │        │
│  └────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. FastAPI Runtime Layer

**Location:** `libs/runtime/cogniverse_runtime/routers/admin.py`

**Responsibilities:**
- HTTP endpoint handling
- Request validation via Pydantic models
- Profile validation via ProfileValidator
- Schema deployment coordination
- Error handling and HTTP responses

**Key Endpoints:**
- `POST /admin/profiles` - Create profile
- `GET /admin/profiles` - List profiles for tenant
- `GET /admin/profiles/{profile_name}` - Get profile details
- `PUT /admin/profiles/{profile_name}` - Update mutable fields
- `DELETE /admin/profiles/{profile_name}` - Delete profile
- `POST /admin/profiles/{profile_name}/deploy` - Deploy schema

**Dependencies:**
- ConfigManager (injected via FastAPI Depends)
- ProfileValidator (injected via FastAPI Depends)
- SchemaLoader (for deployment)

### 2. Configuration Layer

**Location:** `libs/foundation/cogniverse_foundation/config/manager.py`

**ConfigManager:**

Centralized configuration manager with:
- Multi-tenant support
- Thread-safe operations
- Profile CRUD operations
- Deep merge for tenant overrides

**Thread Safety:**
```python
class ConfigManager:
    def __init__(self, store: ConfigStore, cache_size: int = 100):
        self.store = store
        self._backend_lock = threading.Lock()  # Protects read-modify-write

    def add_backend_profile(self, profile, tenant_id, service):
        with self._backend_lock:  # Atomic operation
            backend_config = self.get_backend_config(...)
            backend_config.add_profile(profile)
            self.set_backend_config(backend_config, ...)
```

**Why Locking?**

Without locks, concurrent operations have a race condition:

```python
# Thread 1: Reads config (profiles: A, B)
# Thread 2: Reads config (profiles: A, B)
# Thread 1: Adds profile C → Writes (profiles: A, B, C)
# Thread 2: Adds profile D → Writes (profiles: A, B, D)
# Result: Profile C is LOST!
```

The `_backend_lock` ensures:
1. Thread 1 acquires lock
2. Thread 1 reads, modifies, writes
3. Thread 1 releases lock
4. Thread 2 acquires lock (sees profiles A, B, C)
5. Thread 2 reads, modifies, writes (profiles: A, B, C, D)
6. Both profiles persist correctly

**BackendConfig:**

Pydantic model representing tenant's backend configuration:

```python
class BackendConfig:
    tenant_id: str
    profiles: Dict[str, BackendProfileConfig]  # profile_name → config

    def add_profile(self, profile: BackendProfileConfig):
        """Add or replace profile"""
        self.profiles[profile.profile_name] = profile

    def merge_profile(self, profile_name: str, overrides: Dict[str, Any]):
        """Deep merge overrides into existing profile"""
        base = self.profiles[profile_name]
        return BackendProfileConfig(...merged fields...)
```

### 3. Persistence Layer

**Location:** `libs/foundation/cogniverse_foundation/config/sqlite/config_store.py`

**SQLiteConfigStore:**

Database-backed configuration storage with:
- Multi-tenant isolation
- Version tracking
- WAL mode for concurrency
- Transaction management

**Concurrency Features:**

```python
class SQLiteConfigStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._enable_wal_mode()  # Enable Write-Ahead Logging

    def _enable_wal_mode(self):
        """Enable WAL for better concurrent access"""
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")

    def set_config(self, tenant_id, scope, service, config_key, config_value):
        """Set config with transaction and locking"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            conn.execute("BEGIN IMMEDIATE")  # Exclusive write lock
            cursor = conn.cursor()

            # Get current version
            cursor.execute("""
                SELECT MAX(version) FROM configurations
                WHERE tenant_id=? AND scope=? AND service=? AND config_key=?
            """, (tenant_id, scope, service, config_key))
            current_version = cursor.fetchone()[0] or 0

            # Insert new version
            cursor.execute("""
                INSERT INTO configurations
                (tenant_id, scope, service, config_key, config_value, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """, (tenant_id, scope, service, config_key, json.dumps(config_value), current_version + 1))

            conn.commit()
        except Exception as e:
            conn.rollback()
            raise
        finally:
            conn.close()
```

**WAL Mode Benefits:**
- Readers don't block writers
- Writers don't block readers
- Multiple readers can run concurrently
- Only writers block each other (via BEGIN IMMEDIATE)

**Transaction Types:**
- `BEGIN IMMEDIATE`: Acquires write lock immediately, prevents writer starvation
- Regular `BEGIN`: Acquires lock on first write, can cause deadlocks

### 4. Validation Layer

**Location:** `libs/core/cogniverse_core/validation/profile_validator.py`

**ProfileValidator:**

Validates profile configurations against business rules:

```python
class ProfileValidator:
    def __init__(self, schema_dir: Optional[Path] = None):
        self.schema_dir = schema_dir

    def validate_profile(self, profile: BackendProfileConfig) -> List[str]:
        """Validate complete profile"""
        errors = []
        errors.extend(self._validate_profile_name(profile.profile_name))
        errors.extend(self._validate_schema_exists(profile.schema_name))
        errors.extend(self._validate_embedding_model(profile.embedding_model))
        errors.extend(self._validate_embedding_type(profile.embedding_type))
        return errors

    def validate_update_fields(self, overrides: Dict[str, Any]) -> List[str]:
        """Validate that only mutable fields are being updated"""
        immutable_fields = {
            "profile_name", "type", "schema_name",
            "embedding_model", "embedding_type", "tenant_id"
        }
        errors = []
        for field in overrides.keys():
            if field in immutable_fields:
                errors.append(f"Cannot update immutable field: {field}")
        return errors
```

**Validation Rules:**
- Profile name: alphanumeric + underscore, unique within tenant
- Schema name: must exist in schema directory
- Embedding model: format `org/model` or `model-name`
- Embedding type: enum (`frame_based`, `chunk_based`, `global`)
- Strategies: valid JSON array
- Pipeline config: valid JSON object

### 5. Dashboard UI Layer

**Location:** `scripts/backend_profile_tab.py`

**Streamlit-based UI for profile management:**

```python
def render_backend_profile_tab():
    """Render the backend profile management UI"""
    st.title("Backend Profile Management")

    # Tenant selection
    tenant_id = st.text_input("Tenant ID", value=st.session_state.get("tenant_id", "default"))

    # Profile list
    profiles = list_profiles_from_api(tenant_id)

    # Actions
    if st.button("Create New Profile"):
        show_create_form()

    for profile in profiles:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        col1.write(profile["profile_name"])
        if col2.button("Edit", key=f"edit_{profile['profile_name']}"):
            show_edit_form(profile)
        if col3.button("Deploy", key=f"deploy_{profile['profile_name']}"):
            deploy_schema_via_api(profile["profile_name"], tenant_id)
        if col4.button("Delete", key=f"delete_{profile['profile_name']}"):
            delete_profile_via_api(profile["profile_name"], tenant_id)
```

**API Integration:**

Dashboard uses httpx to call FastAPI endpoints:

```python
def deploy_schema_via_api(profile_name: str, tenant_id: str, force: bool = False):
    """Deploy schema via admin API"""
    api_url = get_runtime_api_url()
    endpoint = f"{api_url}/admin/profiles/{profile_name}/deploy"

    with httpx.Client(timeout=30.0) as client:
        response = client.post(endpoint, json={"tenant_id": tenant_id, "force": force})
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": response.text}
```

## Data Flow

### Profile Creation Flow

```
1. User fills create form in Dashboard
   ↓
2. Dashboard calls POST /admin/profiles via httpx
   ↓
3. Admin router receives request
   ├─ Validates request body (Pydantic)
   ├─ Validates profile fields (ProfileValidator)
   └─ Calls ConfigManager.add_backend_profile()
   ↓
4. ConfigManager acquires _backend_lock
   ├─ Calls get_backend_config() → reads from SQLite
   ├─ Adds profile to BackendConfig in memory
   └─ Calls set_backend_config() → writes to SQLite
   ↓
5. SQLiteConfigStore.set_config()
   ├─ Opens connection with 30s timeout
   ├─ Executes BEGIN IMMEDIATE (write lock)
   ├─ Reads current version
   ├─ Inserts new version
   ├─ Commits transaction
   └─ Closes connection
   ↓
6. ConfigManager releases lock
   ↓
7. Admin router returns 201 Created response
   ↓
8. Dashboard displays success message
```

### Concurrent Updates Flow

```
Thread 1                          Thread 2
   │                                 │
   ├─ update_backend_profile()      │
   ├─ Acquire _backend_lock         │
   │  (LOCKED)                       │
   ├─ get_backend_config()          │
   │  └─ SQLite read (WAL: OK)      ├─ update_backend_profile()
   ├─ Modify in memory              ├─ Try acquire _backend_lock
   ├─ set_backend_config()          │  (BLOCKED - waiting for lock)
   │  └─ BEGIN IMMEDIATE            │
   │  └─ Write version 2            │
   │  └─ COMMIT                      │
   ├─ Release _backend_lock         │
   │                                 ├─ Acquire _backend_lock
   │                                 │  (LOCKED)
   │                                 ├─ get_backend_config()
   │                                 │  └─ SQLite read (sees version 2)
   │                                 ├─ Modify in memory
   │                                 ├─ set_backend_config()
   │                                 │  └─ BEGIN IMMEDIATE
   │                                 │  └─ Write version 3
   │                                 │  └─ COMMIT
   │                                 ├─ Release _backend_lock
   │                                 ▼
```

### Schema Deployment Flow

```
1. User clicks "Deploy" in Dashboard
   ↓
2. Dashboard calls POST /admin/profiles/{name}/deploy
   ↓
3. Admin router receives request
   ├─ Gets profile from ConfigManager
   ├─ Generates tenant schema name: {tenant_id}_{profile_name}
   ├─ Loads schema template from SchemaLoader
   └─ Calls backend deployment API
   ↓
4. Vespa Backend
   ├─ Validates schema structure
   ├─ Creates document type
   ├─ Sets up indexes
   └─ Returns deployment status
   ↓
5. Admin router returns deployment result
   ↓
6. Dashboard updates UI with status
```

## Multi-Tenant Isolation

### Tenant Scoping

Every operation is scoped by `tenant_id`:

```python
# Get profiles for tenant A
profiles_a = config_manager.list_backend_profiles(tenant_id="tenant_a")

# Get profiles for tenant B
profiles_b = config_manager.list_backend_profiles(tenant_id="tenant_b")

# No overlap - complete isolation
```

### Database-Level Isolation

SQLite queries always include `tenant_id`:

```sql
SELECT config_value FROM configurations
WHERE tenant_id = 'tenant_a'
  AND scope = 'backend'
  AND service = 'backend'
  AND config_key = 'backend_config'
ORDER BY version DESC
LIMIT 1;
```

### Schema Name Isolation

Deployed schemas include tenant prefix:

```
tenant_a → profile: video_colpali → schema: tenant_a_video_colpali
tenant_b → profile: video_colpali → schema: tenant_b_video_colpali
```

No naming conflicts in Vespa.

## Version Control

### Version Incrementing

Each `set_config()` creates a new version:

```
Version 1: Initial profile creation
Version 2: Update pipeline_config
Version 3: Update strategies
Version 4: Update description
```

### Version History

Query historical versions:

```python
history = config_manager.get_config_history(
    tenant_id="acme",
    scope=ConfigScope.BACKEND,
    service="backend",
    config_key="backend_config",
    limit=10
)
```

### Optimistic Concurrency

Admin router returns version in update response:

```json
{
  "version": 3,
  "updated_fields": ["pipeline_config"]
}
```

Clients can detect conflicts by comparing versions.

## Deployment Architecture

### Schema Templates

Schema templates stored on disk:

```
data/schemas/
├── video_test_schema.json
├── video_colpali_base_schema.json
└── video_videoprism_base_schema.json
```

### Template Loading

```python
class FilesystemSchemaLoader:
    def load_schema(self, schema_name: str) -> dict:
        """Load schema template from disk"""
        schema_path = self.schema_dir / f"{schema_name}_schema.json"
        with open(schema_path) as f:
            return json.load(f)
```

### Tenant Schema Generation

```python
def generate_tenant_schema(profile: BackendProfileConfig, tenant_id: str) -> dict:
    """Generate tenant-specific schema from template"""
    base_schema = schema_loader.load_schema(profile.schema_name)
    tenant_schema = copy.deepcopy(base_schema)

    # Rename schema to tenant-specific name
    tenant_schema["name"] = f"{tenant_id}_{profile.profile_name}"

    # Apply profile-specific configurations
    # (e.g., embedding dimensions, field types)

    return tenant_schema
```

## Testing Strategy

### Unit Tests

**Location:** `tests/admin/test_profile_*.py`

Test individual components:
- ConfigManager methods
- ProfileValidator rules
- BackendConfig operations

### Integration Tests

**Multi-Tenant Tests** (`test_profile_multi_tenant.py`):
- Tenant isolation
- Cross-tenant access prevention
- Same profile name across tenants

**Concurrent Tests** (`test_profile_concurrent.py`):
- Concurrent profile creation
- Concurrent updates (version tracking)
- Concurrent reads during writes
- Concurrent deletes

**UI Tests** (`test_profile_ui_integration.py`):
- Dashboard API helper functions
- Deploy/delete/status workflows
- Error handling

### Test Fixtures

```python
@pytest.fixture
def test_client(temp_schema_dir, tmp_path):
    """Create test client with isolated database"""
    temp_db = tmp_path / "test_config.db"
    config_manager = create_default_config_manager(db_path=temp_db)

    # Set up test environment
    schema_loader = FilesystemSchemaLoader(temp_schema_dir)
    admin.set_config_manager(config_manager)
    admin.set_schema_loader(schema_loader)

    client = TestClient(app)
    yield client

    # Cleanup
    admin.reset_dependencies()
```

## Performance Characteristics

### Read Performance

- **Cached reads**: O(1) lookup in ConfigManager cache
- **Uncached reads**: O(1) SQLite index lookup
- **List operations**: O(n) where n = number of profiles per tenant

### Write Performance

- **Single write**: ~1-5ms (SQLite local write)
- **Concurrent writes**: Serialized via threading lock, no deadlocks
- **Transaction overhead**: Minimal with WAL mode

### Concurrency Limits

- **SQLite WAL mode**: Unlimited concurrent readers
- **Write throughput**: Limited by threading lock (one writer at a time)
- **Typical load**: 10-100 writes/sec per tenant

### Scaling Considerations

For high concurrency (>1000 writes/sec):
- Consider PostgreSQL instead of SQLite
- Use row-level locking instead of threading locks
- Implement connection pooling

## Security Considerations

### Tenant Isolation

- Strict `tenant_id` validation
- No cross-tenant access possible
- Database queries always filtered by tenant

### Input Validation

- All inputs validated via Pydantic models
- Profile names sanitized (alphanumeric + underscore)
- JSON fields validated for syntax

### SQL Injection Protection

- Parameterized queries only
- No dynamic SQL construction
- SQLite connection with safe defaults

### Future Enhancements

- API key authentication
- Role-based access control (RBAC)
- Audit logging for all operations
- Encryption at rest

## Related Documentation

- [Profile Management Dashboard](../user/profile-management.md) - User guide
- [Profile API Reference](../user/profile-api-reference.md) - API docs
- [Configuration System](../CONFIGURATION_SYSTEM.md) - Overall config architecture
