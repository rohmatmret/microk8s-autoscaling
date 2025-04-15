# tests/test_kubernetes_api.py
import pytest
from unittest.mock import MagicMock, patch, call
from kubernetes.client.rest import ApiException
from agent.kubernetes_api import KubernetesAPI, KubernetesAPIError

@pytest.fixture
def mock_api():
    with patch('kubernetes.config.load_kube_config'), \
         patch('kubernetes.client.AppsV1Api'), \
         patch('kubernetes.client.CoreV1Api'), \
         patch('prometheus_api_client.PrometheusConnect'):
        api = KubernetesAPI(max_pods=5, max_nodes=3)
        api.apps_api = MagicMock()
        api.core_api = MagicMock()
        api.prometheus = MagicMock()
        yield api

def test_get_cluster_state_success(mock_api):
    mock_api._get_pod_count = MagicMock(return_value=3)
    mock_api._get_node_count = MagicMock(return_value=2)
    mock_api._get_cpu_usage = MagicMock(return_value=75.0)
    mock_api._get_memory_usage = MagicMock(return_value=2e9)  # 2GB
    mock_api._get_latency = MagicMock(return_value=0.15)
    
    state = mock_api.get_cluster_state()
    assert state == {
        'pods': 3,
        'nodes': 2,
        'cpu': 75.0,
        'memory': 2e9,
        'latency': 0.15
    }

def test_safe_scale_within_bounds(mock_api):
    mock_api._get_current_replicas = MagicMock(return_value=2)
    mock_api.apps_api.read_namespaced_deployment.return_value.status.ready_replicas = 3
    
    assert mock_api.safe_scale("app", "default", 3) is True
    mock_api._scale_deployment.assert_called_once_with("app", "default", 3)

def test_safe_scale_upper_bound(mock_api):
    mock_api._get_current_replicas = MagicMock(return_value=5)
    assert mock_api.safe_scale("app", "default", 10) is True  # Clamped to 5
    mock_api._scale_deployment.assert_not_called()

def test_scale_verification_timeout(mock_api):
    mock_api.apps_api.read_namespaced_deployment.return_value.status.ready_replicas = 2
    with patch('time.time', side_effect=[0, 301]), \
         patch('time.sleep'):
        with pytest.raises(KubernetesAPIError):
            mock_api.safe_scale("app", "default", 3)

def test_api_error_handling(mock_api):
    mock_api.apps_api.read_namespaced_deployment.side_effect = ApiException(status=500)
    with pytest.raises(KubernetesAPIError):
        mock_api.safe_scale("app", "default", 2)

def test_node_count_excludes_control_plane(mock_api):
    control_plane = MagicMock()
    control_plane.metadata.labels = {'node-role.kubernetes.io/control-plane': ''}
    worker_node = MagicMock()
    worker_node.metadata.labels = {}
    
    mock_api.core_api.list_node.return_value.items = [control_plane, worker_node]
    assert mock_api._get_node_count() == 1

def test_pod_count_error(mock_api):
    mock_api.core_api.list_namespaced_pod.side_effect = ApiException(status=403)
    with pytest.raises(KubernetesAPIError):
        mock_api._get_pod_count("default")

def test_memory_query_failure(mock_api):
    mock_api.prometheus.custom_query.side_effect = Exception("Prometheus error")
    with pytest.raises(KubernetesAPIError):
        mock_api._get_memory_usage("default")

def test_scale_validation(mock_api):
    with pytest.raises(ValueError):
        mock_api._scale_deployment("app", "default", 6)  # Max pods is 5 in fixture