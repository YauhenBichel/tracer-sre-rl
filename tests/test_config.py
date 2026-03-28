"""Tests for YAML config loading."""

from src.config import (
    base_latency_ms,
    crawler_config,
    distractor_diagnoses,
    distractor_remediations,
    log_templates,
    metric_baselines,
)


def test_metric_baselines_loads():
    baselines = metric_baselines()

    assert "web-server" in baselines
    assert "application" in baselines
    assert "database" in baselines
    for svc_type, metrics in baselines.items():
        for name, (low, high) in metrics.items():
            assert low < high, f"{svc_type}.{name}: low ({low}) >= high ({high})"


def test_base_latency_ms_loads():
    latencies = base_latency_ms()
    assert latencies["database"] < latencies["web-server"]
    assert all(v > 0 for v in latencies.values())


def test_log_templates_loads():
    templates = log_templates()
    assert "web-server" in templates
    for svc_type, levels in templates.items():
        assert "ERROR" in levels, f"{svc_type} missing ERROR templates"
        for level, msgs in levels.items():
            assert len(msgs) > 0, f"{svc_type}.{level} has no templates"


def test_distractor_diagnoses():
    distractors = distractor_diagnoses()
    assert len(distractors) > 0
    assert all("." in d for d in distractors)


def test_distractor_remediations():
    remediations = distractor_remediations()
    assert len(remediations) > 0


def test_crawler_config_loads():
    gcp = crawler_config("gcp")

    assert "url" in gcp
    assert "quality_fields" in gcp

    cloudflare = crawler_config("cloudflare")
    assert "url" in cloudflare

    github = crawler_config("github_postmortems")
    assert "url" in github
    assert "category_patterns" in github
