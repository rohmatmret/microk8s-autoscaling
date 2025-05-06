#!/bin/bash

echo "🛑 Stopping simulation processes..."

# Process cleanup
pkill -f dqn.py || true
pkill -f ppo.py || true
pkill -f k6 || true
pkill -f "port-forward" || true

# Kubernetes resource cleanup with force deletion
echo "🧹 Cleaning Kubernetes resources..."
kubectl delete --all all --all-namespaces --force --grace-period=0 2>/dev/null
kubectl delete --all configmaps --all-namespaces --force --grace-period=0 2>/dev/null
kubectl delete --all pvc --all-namespaces --force --grace-period=0 2>/dev/null
kubectl delete --all ingress --all-namespaces --force --grace-period=0 2>/dev/null

# Special handling for observability stack
echo "🔥 Force-cleaning observability stack..."
kubectl patch crds --type=merge -p '{"metadata":{"finalizers":[]}}' 2>/dev/null
for ns in $(kubectl get ns -o name | cut -d/ -f2); do
  kubectl patch namespace $ns --type=json -p='[{"op": "remove", "path": "/metadata/finalizers"}]' 2>/dev/null
done

# MicroK8s specific cleanup
echo "🧼 MicroK8s deep cleanup..."
multipass exec microk8s-vm -- sudo bash -c '
  sudo rm -rf /var/snap/microk8s/common/var/lib/rook/*
  sudo rm -rf /var/snap/microk8s/common/var/lib/kubelet/pods/*
  sudo systemctl restart snap.microk8s.daemon-containerd
'

# Full reset if needed (uncomment if problems persist)
# microk8s reset
# microk8s start

#!/bin/bash

# Daftar namespace yang stuck terminating
# NAMESPACES=$(microk8s kubectl get ns --no-headers | awk '$2=="Terminating"{print $1}')

# if [ -z "$NAMESPACES" ]; then
#   echo "Tidak ada namespace yang dalam status Terminating."
#   exit 0
# fi

# for ns in $NAMESPACES; do
#   echo "Memproses namespace: $ns"

#   # Simpan YAML namespace
#   kubectl get namespace $ns -o json > ${ns}.json

#   # Hapus finalizers dari JSON
#   sed -i '/"finalizers": \[.*\]/d' ${ns}.json
#   sed -i '/"kubernetes"/d' ${ns}.json

#   # Paksa hapus namespace
#   kubectl replace --raw "/api/v1/namespaces/${ns}/finalize" -f ${ns}.json

#   echo "Namespace $ns dipaksa dihapus."
# done


echo "✅ All processes stopped and cluster cleaned."