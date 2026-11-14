"""Read-only MCP server for Kubernetes diagnostics."""

from mcp.server import MCPServer


mcp = MCPServer(
    name="devops-kubernetes-server",
    version="1.0.0",
    instructions="Provides read-only Kubernetes diagnostic tools."
)


# Resources are identified by (namespace, resource_name).

SIMULATED_PODS = {
    ("production", "payment-api"): {
        "name": "payment-api",
        "namespace": "production",
        "status": "Running",
        "ready": "1/1",
        "restarts": 0,
        "node": "worker-node-01",
    },
    ("production", "inventory-api"): {
        "name": "inventory-api",
        "namespace": "production",
        "status": "CrashLoopBackOff",
        "ready": "0/1",
        "restarts": 5,
        "node": "worker-node-02",
    },
    ("staging", "inventory-api"): {
        "name": "inventory-api",
        "namespace": "staging",
        "status": "Running",
        "ready": "1/1",
        "restarts": 0,
        "node": "worker-node-03",
    },
}


SIMULATED_POD_EVENTS = {
    ("production", "payment-api"): [
        {
            "type": "Normal",
            "reason": "Scheduled",
            "message": (
                "Successfully assigned payment-api "
                "to worker-node-01"
            ),
        }
    ],
    ("production", "inventory-api"): [
        {
            "type": "Warning",
            "reason": "BackOff",
            "message": "Back-off restarting failed container",
        },
        {
            "type": "Normal",
            "reason": "Pulled",
            "message": "Container image successfully pulled",
        },
    ],
    ("staging", "inventory-api"): [
        {
            "type": "Normal",
            "reason": "Scheduled",
            "message": (
                "Successfully assigned inventory-api "
                "to worker-node-03"
            ),
        }
    ],
}


SIMULATED_CONTAINER_LOGS = {
    ("production", "payment-api"): [
        "Application started successfully.",
        "Connected to payment database.",
        "Health check passed.",
    ],
    ("production", "inventory-api"): [
        "Starting inventory service...",
        "Connecting to database...",
        "ERROR: Database connection refused.",
        "Application startup failed.",
    ],
    ("staging", "inventory-api"): [
        "Starting inventory service...",
        "Connected to staging database.",
        "Application started successfully.",
    ],
}


SIMULATED_DEPLOYMENTS = {
    ("production", "payment-api"): {
        "name": "payment-api",
        "namespace": "production",
        "desired_replicas": 3,
        "ready_replicas": 3,
        "available_replicas": 3,
        "status": "Healthy",
    },
    ("production", "inventory-api"): {
        "name": "inventory-api",
        "namespace": "production",
        "desired_replicas": 3,
        "ready_replicas": 1,
        "available_replicas": 1,
        "status": "Degraded",
    },
    ("staging", "inventory-api"): {
        "name": "inventory-api",
        "namespace": "staging",
        "desired_replicas": 2,
        "ready_replicas": 2,
        "available_replicas": 2,
        "status": "Healthy",
    },
}


@mcp.tool(
    name="inspect_pod",
    description="Inspect a Kubernetes pod's status and details.",
    structured_output=True,
)
def inspect_pod(
    pod_name: str,
    namespace: str,
) -> dict[str, str | int]:
    """Retrieve information about a pod in a specific namespace."""

    pod = SIMULATED_PODS.get((namespace, pod_name))

    if pod is None:
        return {
            "error": (
                f"Pod '{pod_name}' was not found "
                f"in namespace '{namespace}'."
            )
        }

    return pod


@mcp.tool(
    name="get_pod_events",
    description="Retrieve Kubernetes events associated with a pod.",
    structured_output=True,
)
def get_pod_events(
    pod_name: str,
    namespace: str,
) -> list[dict[str, str]]:
    """Retrieve events for a pod in a specific namespace."""

    return SIMULATED_POD_EVENTS.get(
        (namespace, pod_name),
        [],
    )


@mcp.tool(
    name="get_container_logs",
    description="Retrieve container logs for a Kubernetes pod.",
    structured_output=True,
)
def get_container_logs(
    pod_name: str,
    namespace: str,
) -> list[str]:
    """Retrieve logs for a pod in a specific namespace."""

    return SIMULATED_CONTAINER_LOGS.get(
        (namespace, pod_name),
        ["No logs found for this pod in this namespace."],
    )


@mcp.tool(
    name="check_deployment_status",
    description=(
        "Check the health and replica status of a Kubernetes deployment."
    ),
    structured_output=True,
)
def check_deployment_status(
    deployment_name: str,
    namespace: str,
) -> dict[str, str | int]:
    """Retrieve deployment health in a specific namespace."""

    deployment = SIMULATED_DEPLOYMENTS.get(
        (namespace, deployment_name)
    )

    if deployment is None:
        return {
            "error": (
                f"Deployment '{deployment_name}' was not found "
                f"in namespace '{namespace}'."
            )
        }

    return deployment


if __name__ == "__main__":
    mcp.run(transport="stdio")