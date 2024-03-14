#!/bin/bash

if [ "$USER_TYPE" = "client" ]; then
    # Parse layers argument, generate a list of URLs, and parse the device argument
    IFS=',' read -ra layer_url_mappings <<< "$LAYER_URL_MAPPING"

    # Execute Python script with arguments
    exec python /app/docker/docker_test/client.py --layer_url_mapping "${layer_url_mappings[@]}" --device $DEVICE --model $REPO
elif [ "$USER_TYPE" = "server" ]; then
    # IFS=':' read -ra layers <<< "$LAYERS"
    # layer_ids=($(seq ${layers[0]} ${layers[1]}))

    # Execute Python script with arguments
    exec python /app/docker/docker_test/server.py --layers $LAYERS --device $DEVICE --model $REPO
else
    echo "Invalid USER_TYPE specified"
    exit 1
fi