"""HTTP surface checks for V2 deployment resources."""

from __future__ import annotations

from llama_profile_lab.api import create_app


def test_deployment_routes_are_registered_without_removing_v1(tmp_path) -> None:
    app = create_app(tmp_path / "api.db")
    paths = {route.path for route in app.routes}

    assert "/api/experiments" in paths
    assert "/api/experiments/{experiment_id}/events" in paths
    assert "/api/deployments" in paths
    assert "/api/deployments/{deployment_id}" in paths
    assert "/api/deployments/{deployment_id}/plan" in paths
    assert "/api/deployments/{deployment_id}/candidates" in paths
    assert "/api/deployments/{deployment_id}/placements" in paths
    assert "/api/deployments/{deployment_id}/runs" in paths
    assert "/api/deployments/{deployment_id}/results" in paths
    assert "/api/deployments/{deployment_id}/pareto" in paths
