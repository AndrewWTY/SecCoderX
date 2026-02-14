#!/bin/bash
docker ps -a --filter "name=cweval" --format "{{.ID}}" | xargs -r docker rm -f
