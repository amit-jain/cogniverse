"""
Vespa-based workflow intelligence storage with multi-tenant support.
Stores workflow executions, agent performance, and templates in Vespa backend.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from vespa.application import Vespa

from cogniverse_sdk.interfaces.workflow_store import (
    AgentPerformanceRecord,
    AgentStats,
    ExecutionRecord,
    WorkflowStore,
    WorkflowTemplate,
)

logger = logging.getLogger(__name__)


class VespaWorkflowStore(WorkflowStore):
    """
    Vespa-based workflow store with multi-tenant support.

    Stores workflow data as Vespa documents in a dedicated schema.
    Uses record_type field to distinguish between different record types.

    Schema: workflow_intelligence
    Record types:
    - execution: Workflow execution records
    - performance: Agent performance records
    - template: Workflow templates
    """

    def __init__(
        self,
        vespa_app: Optional[Vespa] = None,
        backend_url: str = "http://localhost",
        backend_port: int = 8080,
        schema_name: str = "workflow_intelligence",
    ):
        """
        Initialize Vespa workflow store.

        Args:
            vespa_app: Existing Vespa application instance (optional)
            backend_url: Backend server URL
            backend_port: Backend server port
            schema_name: Vespa schema name for workflow storage
        """
        if vespa_app is not None:
            self.vespa_app = vespa_app
        else:
            self.vespa_app = Vespa(url=f"{backend_url}:{backend_port}")

        self.schema_name = schema_name
        logger.info(
            f"VespaWorkflowStore initialized with schema: {schema_name} "
            f"at {backend_url}:{backend_port}"
        )

    def initialize(self) -> None:
        """
        Initialize the workflow store.

        For Vespa, this assumes the schema already exists.
        Schema must be deployed separately via vespa-cli or application package.
        """
        try:
            self.vespa_app.query(
                yql=f"select * from {self.schema_name} where true limit 1"
            )
            logger.info(f"Vespa schema '{self.schema_name}' is accessible")
        except Exception as e:
            logger.warning(
                f"Could not verify Vespa schema '{self.schema_name}': {e}. "
                "Ensure schema is deployed before using VespaWorkflowStore."
            )

    def _generate_id(self, prefix: str) -> str:
        """Generate unique document ID."""
        return f"{prefix}:{uuid.uuid4().hex[:12]}"

    def _feed_document(self, doc_id: str, fields: Dict[str, Any]) -> None:
        """Feed a document to Vespa."""
        try:
            response = self.vespa_app.feed_data_point(
                schema=self.schema_name,
                data_id=doc_id,
                fields=fields,
            )
            if response.status_code not in (200, 201):
                raise ValueError(f"Failed to feed document: {response.json}")
        except Exception as e:
            logger.error(f"Failed to feed document {doc_id}: {e}")
            raise

    def _query_documents(
        self,
        yql: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query documents from Vespa."""
        try:
            response = self.vespa_app.query(yql=yql)
            if response.hits:
                return [hit["fields"] for hit in response.hits[:limit]]
            return []
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    # ==================== Execution Records ====================

    def record_execution(
        self,
        tenant_id: str,
        workflow_name: str,
        status: str,
        metrics: Dict[str, Any],
    ) -> str:
        """Record workflow execution."""
        execution_id = self._generate_id("exec")
        now = datetime.utcnow().isoformat()

        fields = {
            "doc_id": execution_id,
            "tenant_id": tenant_id,
            "record_type": "execution",
            "workflow_name": workflow_name,
            "status": status,
            "metrics": json.dumps(metrics),
            "created_at": now,
            "updated_at": now,
        }

        self._feed_document(execution_id, fields)
        logger.debug(f"Recorded execution {execution_id} for workflow {workflow_name}")
        return execution_id

    def get_execution(self, execution_id: str) -> Optional[ExecutionRecord]:
        """Get execution by ID."""
        yql = (
            f"select * from {self.schema_name} "
            f"where doc_id = '{execution_id}' and record_type = 'execution' "
            f"limit 1"
        )

        results = self._query_documents(yql, limit=1)
        if not results:
            return None

        fields = results[0]
        return ExecutionRecord(
            execution_id=fields["doc_id"],
            tenant_id=fields["tenant_id"],
            workflow_name=fields.get("workflow_name", ""),
            status=fields.get("status", ""),
            metrics=json.loads(fields.get("metrics", "{}")),
            created_at=datetime.fromisoformat(fields["created_at"]),
            updated_at=datetime.fromisoformat(fields["updated_at"]),
        )

    def list_executions(
        self,
        tenant_id: str,
        workflow_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[ExecutionRecord]:
        """List executions with optional filters."""
        conditions = [
            f"tenant_id = '{tenant_id}'",
            "record_type = 'execution'",
        ]
        if workflow_name:
            conditions.append(f"workflow_name = '{workflow_name}'")

        yql = (
            f"select * from {self.schema_name} "
            f"where {' and '.join(conditions)} "
            f"order by created_at desc limit {limit}"
        )

        results = self._query_documents(yql, limit=limit)
        executions = []
        for fields in results:
            executions.append(
                ExecutionRecord(
                    execution_id=fields["doc_id"],
                    tenant_id=fields["tenant_id"],
                    workflow_name=fields.get("workflow_name", ""),
                    status=fields.get("status", ""),
                    metrics=json.loads(fields.get("metrics", "{}")),
                    created_at=datetime.fromisoformat(fields["created_at"]),
                    updated_at=datetime.fromisoformat(fields["updated_at"]),
                )
            )
        return executions

    # ==================== Agent Performance ====================

    def record_agent_performance(
        self,
        tenant_id: str,
        agent_type: str,
        duration_ms: float,
        success: bool,
        metrics: Dict[str, Any],
    ) -> str:
        """Record agent performance for a single execution."""
        performance_id = self._generate_id("perf")
        now = datetime.utcnow().isoformat()

        # Include duration and success in metrics for storage
        full_metrics = {
            **metrics,
            "duration_ms": duration_ms,
            "success": success,
        }

        fields = {
            "doc_id": performance_id,
            "tenant_id": tenant_id,
            "record_type": "performance",
            "agent_type": agent_type,
            "metrics": json.dumps(full_metrics),
            "created_at": now,
            "updated_at": now,
        }

        self._feed_document(performance_id, fields)
        logger.debug(f"Recorded performance {performance_id} for agent {agent_type}")
        return performance_id

    def get_agent_stats(
        self,
        tenant_id: str,
        agent_type: str,
    ) -> Optional[AgentStats]:
        """Get aggregated agent statistics."""
        yql = (
            f"select * from {self.schema_name} "
            f"where tenant_id = '{tenant_id}' "
            f"and record_type = 'performance' "
            f"and agent_type = '{agent_type}' "
            f"order by created_at desc limit 1000"
        )

        results = self._query_documents(yql, limit=1000)
        if not results:
            return None

        # Aggregate statistics
        total = len(results)
        total_duration = 0.0
        success_count = 0
        last_execution = None

        for fields in results:
            metrics = json.loads(fields.get("metrics", "{}"))
            total_duration += metrics.get("duration_ms", 0.0)
            if metrics.get("success", True):
                success_count += 1
            if last_execution is None:
                last_execution = datetime.fromisoformat(fields["created_at"])

        return AgentStats(
            agent_type=agent_type,
            tenant_id=tenant_id,
            total_executions=total,
            avg_duration_ms=total_duration / total if total > 0 else 0.0,
            success_rate=success_count / total if total > 0 else 0.0,
            last_execution=last_execution,
        )

    def list_agent_performance(
        self,
        tenant_id: str,
        agent_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[AgentPerformanceRecord]:
        """List agent performance records."""
        conditions = [
            f"tenant_id = '{tenant_id}'",
            "record_type = 'performance'",
        ]
        if agent_type:
            conditions.append(f"agent_type = '{agent_type}'")

        yql = (
            f"select * from {self.schema_name} "
            f"where {' and '.join(conditions)} "
            f"order by created_at desc limit {limit}"
        )

        results = self._query_documents(yql, limit=limit)
        records = []
        for fields in results:
            metrics = json.loads(fields.get("metrics", "{}"))
            records.append(
                AgentPerformanceRecord(
                    performance_id=fields["doc_id"],
                    tenant_id=fields["tenant_id"],
                    agent_type=fields.get("agent_type", ""),
                    duration_ms=metrics.get("duration_ms", 0.0),
                    success=metrics.get("success", True),
                    metrics=metrics,
                    created_at=datetime.fromisoformat(fields["created_at"]),
                )
            )
        return records

    # ==================== Workflow Templates ====================

    def save_template(
        self,
        tenant_id: str,
        template_name: str,
        config: Dict[str, Any],
    ) -> str:
        """Save workflow template (create or update)."""
        # Check if template already exists
        existing = self.get_template(tenant_id, template_name)
        now = datetime.utcnow().isoformat()

        if existing:
            template_id = existing.template_id
            created_at = existing.created_at.isoformat()
        else:
            template_id = self._generate_id("tmpl")
            created_at = now

        fields = {
            "doc_id": template_id,
            "tenant_id": tenant_id,
            "record_type": "template",
            "workflow_name": template_name,  # Reuse field for template name
            "template_config": json.dumps(config),
            "created_at": created_at,
            "updated_at": now,
        }

        self._feed_document(template_id, fields)
        logger.debug(f"Saved template {template_id}: {template_name}")
        return template_id

    def get_template(
        self,
        tenant_id: str,
        template_name: str,
    ) -> Optional[WorkflowTemplate]:
        """Get template by name."""
        yql = (
            f"select * from {self.schema_name} "
            f"where tenant_id = '{tenant_id}' "
            f"and record_type = 'template' "
            f"and workflow_name = '{template_name}' "
            f"limit 1"
        )

        results = self._query_documents(yql, limit=1)
        if not results:
            return None

        fields = results[0]
        return WorkflowTemplate(
            template_id=fields["doc_id"],
            tenant_id=fields["tenant_id"],
            template_name=fields.get("workflow_name", ""),
            config=json.loads(fields.get("template_config", "{}")),
            created_at=datetime.fromisoformat(fields["created_at"]),
            updated_at=datetime.fromisoformat(fields["updated_at"]),
        )

    def list_templates(self, tenant_id: str) -> List[WorkflowTemplate]:
        """List all templates for tenant."""
        yql = (
            f"select * from {self.schema_name} "
            f"where tenant_id = '{tenant_id}' and record_type = 'template' "
            f"order by workflow_name asc limit 1000"
        )

        results = self._query_documents(yql, limit=1000)
        templates = []
        for fields in results:
            templates.append(
                WorkflowTemplate(
                    template_id=fields["doc_id"],
                    tenant_id=fields["tenant_id"],
                    template_name=fields.get("workflow_name", ""),
                    config=json.loads(fields.get("template_config", "{}")),
                    created_at=datetime.fromisoformat(fields["created_at"]),
                    updated_at=datetime.fromisoformat(fields["updated_at"]),
                )
            )
        return templates

    def delete_template(
        self,
        tenant_id: str,
        template_name: str,
    ) -> bool:
        """Delete a template."""
        template = self.get_template(tenant_id, template_name)
        if not template:
            return False

        try:
            response = self.vespa_app.delete_data(
                schema=self.schema_name,
                data_id=template.template_id,
            )
            if response.status_code in (200, 204):
                logger.debug(f"Deleted template: {template_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete template {template_name}: {e}")
            return False

    # ==================== Utility Methods ====================

    def health_check(self) -> bool:
        """Check if storage backend is healthy."""
        try:
            self.vespa_app.query(
                yql=f"select * from {self.schema_name} where true limit 1"
            )
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "executions": 0,
            "performance_records": 0,
            "templates": 0,
        }

        for record_type in ["execution", "performance", "template"]:
            yql = (
                f"select * from {self.schema_name} "
                f"where record_type = '{record_type}' limit 0"
            )
            try:
                response = self.vespa_app.query(yql=yql)
                if hasattr(response, "number_documents_retrieved"):
                    if record_type == "execution":
                        stats["executions"] = response.number_documents_retrieved
                    elif record_type == "performance":
                        count = response.number_documents_retrieved
                        stats["performance_records"] = count
                    else:
                        stats["templates"] = response.number_documents_retrieved
            except Exception:
                pass

        return stats
