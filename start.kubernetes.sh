#!/bin/bash

set -euo pipefail

sudo kubectl apply -f k3s/01-namespace.yaml
sudo kubectl apply -f k3s/
