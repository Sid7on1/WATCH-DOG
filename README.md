# WATCHDOG: Multi-Agent AI/ML Research Paper Processing System

Advanced multi-agent pipeline that discovers, analyzes, and implements AI/ML research papers into production code.

## Focus Areas

1. RAG Systems & Vector Databases
2. Advanced NLP & Language Models
3. AI Agents & Multi-Agent Systems
4. MLOps/CI-CD & Deployment
5. Computer Vision & Neural Networks

## Architecture

scraper -> extractor -> selector -> planner -> manager -> 4 coding agents

## Coding Agents

- Coder 1: RAG & Vector DB Specialist
- Coder 2: NLP & LLM Specialist
- Coder 3: AI Agents Specialist
- Coder 4: MLOps & Computer Vision Specialist

## Installation

pip install -r requirements.txt

## Configuration

Required API keys in .env:
- OPEN_API or OPENROUTER_API_KEY
- groq_API, cohere_API
- GEMINI_API_KEY (optional)
- GITHUB_PAT

## Usage

python enhanced_watchdog.py

## License

See LICENSE file.
