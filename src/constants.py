"""Shared constants used across the codebase.

Any string or value that appears in more than one file belongs here.
"""

from __future__ import annotations

# Taxonomy path separator
TAXONOMY_SEPARATOR = "."

# Event types
EVENT_ALERT = "alert"
EVENT_CASCADE = "cascade"

# Event type groups
SYSTEM_EVENT_TYPES = frozenset({"alert", "deployment", "scaling", "config_change"})
ERROR_EVENT_TYPES = frozenset({"error_spike", "cascade", "connection_exhaustion", "resource_exhaustion"})
WARN_EVENT_TYPES = frozenset({"traffic_ramp", "latency_spike", "symptom", "memory_leak"})

# Service types
SERVICE_TYPE_WEB_SERVER = "web-server"
SERVICE_TYPE_APPLICATION = "application"

# Metric name substrings used for matching in effect handlers
METRIC_LATENCY = "latency"
METRIC_ERROR_RATE = "error_rate"
METRIC_CONNECTION = "connection"
METRIC_PERCENT = "percent"
METRIC_CPU_PERCENT = "cpu_percent"
METRIC_MEMORY_PERCENT = "memory_percent"
METRIC_DISK_USAGE_PERCENT = "disk_usage_percent"
METRIC_REQUEST_RATE = "request_rate"
METRIC_QUERY_RATE = "query_rate"
METRIC_ENQUEUE_RATE = "enqueue_rate"

# Alert metadata keys
META_ALERT_NAME = "alert_name"
META_SEVERITY = "severity"
META_SEVERITY_UNKNOWN = "unknown"
