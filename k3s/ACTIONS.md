# YOLO Service k3s Notes

This directory deploys only the C++ YOLO/WebRTC worker for this repo.

The frontend and signaling server keep their own k3s manifests in their own repositories.
The shared contract between the three repos is:

- namespace: `cam-det`
- signaling service DNS: `signaling-app.cam-det.svc.cluster.local`
- signaling WebSocket path: `/ws`
- shared JWT secret between `signaling-app` and `yolo-service`

## Build Image

Build or import the YOLO image into the k3s container runtime before applying:

```bash
docker build -t docker.io/library/yolo-service:v1 /home/theyuhur/Projects/C-_camera_yolo_service
```

If k3s uses containerd instead of Docker directly, import the images:

```bash
docker save docker.io/library/yolo-service:v1 | sudo k3s ctr images import -
```

## Apply

```bash
./start.kubernetes.sh
```

## Stop

```bash
./stop.kubernetes.sh
```

## Check Status

```bash
sudo kubectl get pods,svc,ingress,pvc -n cam-det
sudo kubectl logs -n cam-det deploy/yolo-service
```

## Shared Secrets

The JWT secret must match between `signaling-app-secret` and `yolo-service-secret`.
Rotate the checked-in values before using this outside a private local cluster.
