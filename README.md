# llm-fission

**Efficiently Distribute Large Language Models (LLM) among Multiple Machines**

## Overview

This repository, named `llm-fission`, addresses the need to distribute Large Language Models efficiently across multiple machines. The system is designed to facilitate the execution of substantial language models by distributing the workload among various machines.

## Components

### Server
A server is a machine responsible for hosting one or multiple transformer decoder layers.

### Client
A client is a machine responsible for hosting the embedding layer and lm_head.

## System Architecture
#### Server
*Servers* host one or multiple transformer decoder layers within the distributed environment. Each server listens for incoming requests from clients at its designated port.

##### Building and Running a Server Container
1. Begin by creating a dedicated Docker network:
   ```bash
   docker network create mynetwork
   ```

2. Construct a Docker image based on the provided Dockerfile:
   ```bash
   docker build -t docker_test -f <path/to/dockerfile> .
   ```

3. Launch a server container, designating its role ('server') alongside the required layer range (0 through 11):
   ```bash
   docker run -e DEVICE="cpu" -e REPO="Writer/palmyra-small" -e USER_TYPE=server -e LAYERS=0:11 --network my_network -p 7001:7001 --name my_container docker_test
   ```

#### Client
*Clients*, which manage the embedding layer and lm_head, communicate with appropriate servers to fetch processed data seamlessly.

##### Building and Running a Client Container
1. Use the previously created Docker network when building another container:
   ```bash
   docker build -t docker_test -f <path/to/dockerfile> .
   ```

2. Initialize a client container, identifying itself accordingly ('client'), along with the URL mappings associated with each respective server layer:
   ```bash
   docker run -e DEVICE="cpu" -e REPO="Writer/palmyra-small" -e USER_TYPE=client -e LAYER_URL_MAPPING="0:11=http://my_container:7001/block2," --network my_network -p 7002:7002 --name my_container2 docker_test
   ```

## Local Network Setup
For locally accessible machines, establish a shared Docker network called `mynetwork` via the following command:

```bash
docker network create mynetwork
```

## Docker Commands Reference
| Command                          | Description                               | Example           |
|----------------------------------|-------------------------------------------|-------------------|
| `docker network create mynetwork` | Creates a custom Docker network            | See previous note. |
| `docker build ...`              | Generates a Docker image based on a Dockerfile  | `docker build -t...`|
| `docker run ...`                | Executes a Docker container                | Various examples. |

Feel free to reach out if further clarification is needed. Happy coding! 💻
```
