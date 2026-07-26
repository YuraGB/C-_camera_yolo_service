#!/bin/bash

set -euo pipefail

sudo kubectl delete -f k3s/ --ignore-not-found
