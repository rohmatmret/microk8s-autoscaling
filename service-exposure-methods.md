# Kubernetes Service Exposure Methods: NodePort vs. LoadBalancer vs. Ingress

## Introduction
When deploying applications on Kubernetes, exposing services to external traffic is a critical requirement. MicroK8s, the lightweight Kubernetes distribution, offers three primary methods: **NodePort**, **LoadBalancer (via MetalLB)**, and **Ingress**. This article explains each method's use cases, implementation, and tradeoffs.

---

## 1. NodePort: The Simplest Exposure Method

### What is NodePort?
- Exposes services on static ports (30000-32767) across all cluster nodes
- Direct node IP access without DNS requirements

### Implementation
```bash
microk8s kubectl expose deployment nginx --type=NodePort --port=80
```

## Characteristics
|Feature|	Details|
|---|---|
|Access Pattern|	http://<node-ip>:<30000-32767>|
Networking Layer	| L4 (TCP/UDP)|
Setup Complexity|	Minimal|

Best for: Local development and quick testing

2. LoadBalancer (MetalLB): Production-Grade Exposure
What is LoadBalancer?
Assigns dedicated external IPs from a configured pool

Requires MetalLB addon for bare-metal environments

### Implementation
```bash
microk8s enable metallb
microk8s kubectl expose deployment nginx --type=LoadBalancer --port=80
```
## Characteristics
|Feature|	Details|
|---|---|
|Access Pattern|	http://<external-ip>|
|Networking Layer|	L4 (TCP/UDP)|
|Setup Complexity|	Moderate|
Best for: Production environments with static IP requirements
3. Ingress: Advanced Routing and Load Balancing
What is Ingress?
- Manages external access to services via HTTP/S
- Provides advanced routing, SSL termination, and load balancing
- Requires an Ingress controller (e.g., NGINX, Traefik)

### Implementation
```bash
microk8s enable ingress
```
Sample Ingress Resource
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nginx-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
    - host: example.com
      http:
        paths:
        - path: /
        pathType: Prefix
        backend:
          service:
            name: nginx
            port: 
              number: 80
```

### Apply the Ingress Resource
```bash
microk8s kubectl apply -f ingress.yaml

```
### Characteristics
|Feature|	Details|
|---|---|
|Access Pattern|	http://<ingress-ip>/<path>|
|Networking Layer|	L7 (HTTP/S)|
|Setup Complexity|	High|

Best for: Complex applications requiring advanced routing and SSL termination
---


# Comparative Analysis
|Criteria|	NodePort	|LoadBalancer	|Ingress|
|---|---|---|---|
IP Usage|	Node IPs|	Dedicated IP|	Shared IP|
|Access Method|	Node IP + Port|	Dedicated IP|	Host/path-based|
|Health Checks|	None|	None|	Built-in|
|Ports	|30000-32767	|Standard ports|	Standard ports|
|TLS|	Manual config|	Manual config|	Built-in support|
|Routing|	None|	None|	Host/path-based|

Cost	Free	Free (MetalLB)	Free