# llm-fission

**Efficiently Distribute Large Language Models (LLM) among Multiple Machines**

## Overview

This repository, named `llm-fission`, addresses the need to distribute Large Language Models efficiently across multiple machines. The system is designed to facilitate the execution of substantial language models by distributing the workload among various machines.

## Components

### Server
A server is a machine responsible for hosting one or multiple transformer decoder layers.

### Client
A client is a machine responsible for hosting the embedding layer and lm_head.

## Local Network Setup

For machines available on the local network, create a Docker network named `mynetwork` using the following command:

```bash
docker network create mynetwork
