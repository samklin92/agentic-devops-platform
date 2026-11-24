"""Read-only MCP server for Kubernetes diagnostics and service metrics."""

from mcp.server import MCPServer


mcp = MCPServer(
    name="devops-kubernetes-server",
    version="1.0.0",
    instructions=(
        "Provides read-only Kubernetes diagnostic tools and "
        "read-only service metrics tools."
    ),
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


# Service metrics, identified by (namespace, service). Values are the
# simulated state over whatever window the caller asks for.

MIN_WINDOW_MINUTES = 1
MAX_WINDOW_MINUTES = 1440

SIMULATED_ERROR_RATES = {
    ("production", "payment-api"): {"error_rate_pct": 0.2},
    ("production", "inventory-api"): {"error_rate_pct": 98.0},
    ("staging", "inventory-api"): {"error_rate_pct": 0.1},
    ("production", "checkout-api"): {"error_rate_pct": 2.1},
}

SIMULATED_LATENCY = {
    ("production", "payment-api"): {
        "p50_ms": 45, "p95_ms": 120, "p99_ms": 210,
    },
    ("production", "inventory-api"): {
        "p50_ms": 800, "p95_ms": 5000, "p99_ms": 8000,
    },
    ("staging", "inventory-api"): {
        "p50_ms": 40, "p95_ms": 90, "p99_ms": 160,
    },
    ("production", "checkout-api"): {
        "p50_ms": 600, "p95_ms": 2400, "p99_ms": 4800,
    },
}

SIMULATED_RESOURCE_USAGE = {
    ("production", "payment-api"): {
        "cpu_pct_of_limit": 35, "memory_pct_of_limit": 40,
    },
    ("production", "inventory-api"): {
        "cpu_pct_of_limit": 5, "memory_pct_of_limit": 10,
    },
    ("staging", "inventory-api"): {
        "cpu_pct_of_limit": 20, "memory_pct_of_limit": 30,
    },
    ("production", "checkout-api"): {
        "cpu_pct_of_limit": 96, "memory_pct_of_limit": 61,
    },
}


def _window_error(window_minutes: int) -> dict[str, str] | None:
    """Return an error result if the window is not a bounded positive int."""
    if (
        isinstance(window_minutes, bool)
        or not isinstance(window_minutes, int)
        or not MIN_WINDOW_MINUTES <= window_minutes <= MAX_WINDOW_MINUTES
    ):
        return {
            "error": (
                "window_minutes must be an integer between "
                f"{MIN_WINDOW_MINUTES} and {MAX_WINDOW_MINUTES}."
            )
        }
    return None


def _service_not_found(service: str, namespace: str) -> dict[str, str]:
    return {
        "error": (
            f"Service '{service}' was not found "
            f"in namespace '{namespace}'."
        )
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


@mcp.tool(
    name="get_error_rate",
    description="Get a service's error rate over a recent time window.",
    structured_output=True,
)
def get_error_rate(
    service: str,
    namespace: str,
    window_minutes: int = 15,
) -> dict[str, str | int | float]:
    """Retrieve the error rate for a service in a specific namespace."""

    window_error = _window_error(window_minutes)
    if window_error is not None:
        return window_error

    data = SIMULATED_ERROR_RATES.get((namespace, service))
    if data is None:
        return _service_not_found(service, namespace)

    return {
        "service": service,
        "namespace": namespace,
        "window_minutes": window_minutes,
        **data,
    }


@mcp.tool(
    name="get_latency_percentiles",
    description=(
        "Get a service's p50, p95 and p99 latency over a recent time window."
    ),
    structured_output=True,
)
def get_latency_percentiles(
    service: str,
    namespace: str,
    window_minutes: int = 15,
) -> dict[str, str | int | float]:
    """Retrieve latency percentiles for a service in a specific namespace."""

    window_error = _window_error(window_minutes)
    if window_error is not None:
        return window_error

    data = SIMULATED_LATENCY.get((namespace, service))
    if data is None:
        return _service_not_found(service, namespace)

    return {
        "service": service,
        "namespace": namespace,
        "window_minutes": window_minutes,
        **data,
    }


@mcp.tool(
    name="get_resource_usage",
    description=(
        "Get a service's current CPU and memory usage as a percentage "
        "of its limits."
    ),
    structured_output=True,
)
def get_resource_usage(
    service: str,
    namespace: str,
) -> dict[str, str | int | float]:
    """Retrieve resource usage for a service in a specific namespace."""

    data = SIMULATED_RESOURCE_USAGE.get((namespace, service))
    if data is None:
        return _service_not_found(service, namespace)

    return {
        "service": service,
        "namespace": namespace,
        **data,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")