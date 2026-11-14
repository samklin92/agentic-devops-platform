# Agentic DevOps Platform

**An AI-powered, supervisor-mediated multi-agent platform for intelligent DevOps operations and cloud infrastructure management.**

## Overview

Agentic DevOps Platform is an evolving AI engineering project designed to investigate, analyse and support the resolution of complex DevOps and cloud infrastructure incidents.

The platform combines Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), agent orchestration, Model Context Protocol (MCP) and specialist AI agents.

Its long-term architecture is designed around a Supervisor-Mediated Multi-Agent System, where specialist agents investigate different aspects of an operational incident and return structured findings to a central Supervisor.

The project prioritises evidence-based reasoning, controlled tool execution, measurable performance and human approval for consequential infrastructure changes.

## Current Implementation

The existing foundation includes:

- **LLM Integration:** Claude API integration for AI-powered reasoning and report generation.
- **RAG:** Retrieval of relevant DevOps operational knowledge.
- **Vector Search:** Sentence Transformers embeddings and FAISS-based retrieval.
- **Reranking:** Cross-encoder-based relevance reranking.
- **Agent Architecture:** Controlled DevOps agent with explicit execution limits and decision-making logic.
- **LangGraph:** Graph-based agent orchestration, state management and conditional execution.
- **MCP:** Model Context Protocol server exposing Kubernetes diagnostic tools.
- **Checkpointing:** Persistence of agent execution state.
- **Testing:** An established automated test suite.

## Existing Kubernetes Diagnostic Capabilities

The MCP server exposes diagnostic tools including:

- `inspect_pod`
- `get_pod_events`
- `get_container_logs`
- `check_deployment_status`

These capabilities provide the foundation for our planned Kubernetes specialist agent.

## Target Architecture

The planned architecture follows a Supervisor-Mediated Multi-Agent approach.

Initial specialist agents include:

1. Supervisor Agent
2. Kubernetes Troubleshooting Agent
3. Terraform / Infrastructure Agent
4. Log Analysis Agent
5. Cloud Cost Optimisation Agent
6. Security Agent
7. CI/CD Agent
8. Knowledge and RAG Agent
9. Reporting Agent

Agents will communicate through validated, structured contracts.

The Supervisor will correlate findings, request additional investigation when necessary and coordinate recommendations.

Production infrastructure modifications will require human approval.

## Development Approach

The platform is initially being developed as a modular monolith within a single GitHub repository.

Mature specialist agents may eventually be extracted into independent repositories and containerised services.

This phased approach allows us to validate agent responsibilities, communication contracts, reliability and performance before introducing distributed-system complexity.

## Technology Stack

- Python 3.12
- Pydantic
- LangGraph
- LangChain
- Anthropic Claude API
- Model Context Protocol (MCP)
- Sentence Transformers
- FAISS
- Cross-Encoder reranking
- Pytest
- SQLite

## Evaluation Strategy

A major engineering objective is to measure whether the multi-agent architecture improves upon the existing single-agent implementation.

Evaluation will consider:

- Incident diagnosis accuracy.
- Evidence quality.
- Root-cause identification.
- Recommendation quality.
- Tool execution efficiency.
- Latency.
- Token consumption and API cost.
- Reliability and error handling.

The existing single-agent implementation will serve as our baseline.

## Cloud Deployment Strategy

AWS is the intended cloud deployment platform.

Development and initial validation will take place locally.

The eventual demonstration deployment will target a monthly infrastructure budget of approximately $50–$100, with appropriate cost controls and billing alerts.

## Project Status

**Current phase:** Stage 9 — Multi-Agent Architecture.

Completed foundations:

- LLM fundamentals and API integration.
- Function calling and agent loops.
- Agent memory concepts.
- Retrieval-Augmented Generation.
- Agent architecture.
- LangGraph.
- Model Context Protocol.
- Existing controlled DevOps agent.

Current development objectives:

- Design Supervisor and Kubernetes Agent interaction.
- Implement Pydantic communication contracts.
- Implement specialist agent responsibilities.
- Establish multi-agent evaluation against the existing single-agent baseline.

## Engineering Philosophy

The objective is not to introduce multiple agents simply for architectural complexity.

Every specialist agent must have a clearly defined responsibility, measurable contribution and appropriate execution boundaries.

The platform prioritises:

- Evidence before recommendations.
- Controlled execution over unrestricted autonomy.
- Supervisor-mediated coordination.
- Human approval for consequential infrastructure changes.
- Measurable engineering outcomes.

---

**Agentic DevOps Platform — Building intelligent, evidence-driven cloud operations.**