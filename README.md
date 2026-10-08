# PromptRouter Engine

An intelligent orchestration engine built with FastAPI, LangGraph, and LangChain for dynamic, cost-effective, and low-latency routing of LLM tasks.

## Overview
This engine analyzes incoming prompts and dynamically routes them to the most appropriate Large Language Model (e.g., cheaper/faster models for simple tasks, more capable models for complex reasoning tasks).

## Features (In Progress)
- [ ] Phase 1: FastAPI Foundation & Pydantic Models
- [ ] Phase 2: Unified LLM Interface (LangChain) & Basic Routing
- [ ] Phase 3: Intelligent Agentic Routing (LangGraph)
- [ ] Phase 4: Metrics, Cost, and Latency Tracking
- [ ] Phase 5: Validation against 10k+ row dataset

## Setup

1. Activate the virtual environment:
   ```bash
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
