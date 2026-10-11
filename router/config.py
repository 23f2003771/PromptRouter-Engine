import os
from dotenv import load_dotenv

# load environment variables from my .env file
load_dotenv()

# my API keys
TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")

# model mapped to each task type
MODEL_MAP = {
    # 1. Long-Horizon Frontiers & Heavy Automation
    "frontier_autonomous": "gpt-6-astra",          # Top-tier execution for hyper-complex multi-step tasks
    "agentic_orchestration": "claude-5-1-fable",   # Advanced logical orchestration and planning
    
    # 2. Advanced Reasoning & Complex Coding (Chain-of-Thought)
    "reasoning_heavy": "deepseek-r1",              # Best specialized math/logic reasoning engine
    "reasoning_premium": "o3-reasoning",           # OpenAI's top-tier agentic reasoning model
    "reasoning_expert": "claude-5-5-opus",         # Deep premium analysis and research engine
    
    # 3. Flagship Multimodal & Production Workhorses
    "code": "claude-5-sonnet",                     # Gold standard for engineering, complex architecture
    "creative": "claude-5-sonnet",                 # Best-in-class nuanced prose and system workflows
    "factual_long_context": "gemini-3.1-pro",      # Destroys 1M+ token RAG and massive document analysis
    "general_premium": "gpt-5.6-terra",            # Balanced flagship for robust standard API tasks
    "agentic_professional": "gpt-6.1-sol",         # OpenAI's professional automation workhorse

    # 4. High-Throughput & Cost-Optimized (Flash/Mini Tier)
    "agentic_fast": "gemini-3.8-flash",            # Blazing fast, low-latency agentic loops
    "general_budget": "gpt-5.6-luna",              # Fast OpenAI fallback for structured JSON extraction
    
    # 5. Rock-Bottom Baselines & Open Weights
    "factual": "deepseek-v4-flash",                # Incredible baseline knowledge per dollar
    "fallback_cheapest": "gemini-2.5-flash-lite"   # Absolute floor pricing default fallback
}


# estimated cost per 1k tokens (blended input/output for quick math)
# useful to track how much money I save by avoiding expensive models on simple tasks
MODEL_COSTS_PER_1K = {
    # Ultra-Premium / Long-Horizon Frontiers ($30 - $60 blended per 1M tokens)
    "gpt-6-astra": 0.0300,         # Blended from $10.00 In / $50.00 Out
    "claude-5-1-fable": 0.0300,    # Blended from $10.00 In / $50.00 Out
    
    # Premium Reasoning & Specialized Tier ($12 - $20 blended per 1M tokens)
    "claude-5-5-opus": 0.0120,     # Blended from $4.00 In / $20.00 Out
    "gpt-6.1-sol": 0.0060,         # Blended from $2.00 In / $10.00 Out
    "o3-reasoning": 0.0050,        # Blended from $2.00 In / $8.00 Out
    "deepseek-r1": 0.0012,         # Highly disruptive CoT price (~$0.56 In / $1.68 Out)
    
    # Flagship Tier ($4 - $7 blended per 1M tokens)
    "claude-5-sonnet": 0.0060,     # Blended from $3.00 In / $15.00 Out
    "gpt-5.6-terra": 0.0070,       # Blended from $2.00 In / $12.00 Out
    "gemini-3.1-pro": 0.0070,      # Blended from $2.00 In / $12.00 Out (below 200k window)
    
    # Performance Flash/Mini Tier ($0.80 - $1.50 blended per 1M tokens)
    "gemini-3.8-flash": 0.00225,   # Blended from $0.75 In / $3.75 Out
    "gpt-5.6-luna": 0.00070,       # Blended from $0.20 In / $1.20 Out
    
    # Extreme Budget & Open-Weights Tier (< $0.40 blended per 1M tokens)
    "deepseek-v4-flash": 0.00021,  # Blended from $0.14 In / $0.28 Out
    "gemini-2.5-flash-lite": 0.00025 # Blended from $0.10 In / $0.40 Out (Cheapest baseline)
}


# criteria I pass to TypeSafe SDK to classify incoming prompts
CLASSIFICATION_CRITERIA = {
    "frontier_autonomous":
        "HYPER-COMPLEX AUTOMATION: Multi-hour autonomous execution, highly unpredictable environments, "
        "or multi-layered architectural planning requiring long-horizon strategy. "
        "Use for mission-critical industrial orchestration where errors compound disaster. "
        "Examples: 'Spin up a multi-region cloud infrastructure, deploy our app, monitor for traffic spikes, and auto-patch vulnerabilities over the next 6 hours' or 'Act as an autonomous DevOps agent to find and fix memory leaks across our entire production fleet without human intervention.'",

    "agentic_orchestration":
        "ADVANCED LOGICAL PLANNING: Cross-platform task routing, enterprise-level workflow delegation, "
        "and managing stateful, multi-agent frameworks with competing objective functions. "
        "Examples: 'Coordinate a 5-agent team to scrape competitor data, generate localized ad copy, verify compliance, and publish to Shopify, Salesforce, and Twitter' or 'Build a dynamic supply chain routing agent that balances vendor reliability against fluctuating fuel costs across 12 countries.'",

    "reasoning_heavy": 
        "HIGH COMPLEXITY MATH/LOGIC: Requires extended internal chain-of-thought. "
        "Use for multi-step math proofs, deep algorithmic architecture, execution-critical logic, or complex debugging. "
        "Do NOT use for basic coding or standard troubleshooting. "
        "Examples: 'Formally prove the convergence of this custom reinforcement learning loss function' or 'Debug a race condition in a multi-threaded Rust consensus algorithm and provide the exact mathematical correction.'",
        
    "reasoning_premium": 
        "HIGH-STAKES AGENTIC: Autonomous agent actions, multi-file codebase refactoring, or zero-shot complex system orchestration. "
        "Use when flawlessly navigating competing constraints is required. "
        "Examples: 'Refactor our legacy monolithic backend into decoupled microservices while preserving all database schemas' or 'Review this 50-file Git repository and generate a unified migration plan to upgrade from Python 3.9 to 3.12.'",

    "reasoning_expert":
        "PREMIUM ANALYSIS & DEEP RESEARCH: Deep-dive academic synthesis, heavy regulatory compliance cross-referencing, "
        "and evaluating edge-case legal/financial theories requiring massive domain authority. "
        "Examples: 'Cross-reference this 300-page medical trial data against FDA Phase-3 guidelines and highlight all compliance vulnerabilities' or 'Draft a comprehensive legal defense strategy analyzing the conflicting precedents in these corporate tax law cases.'",
        
    "code": 
        "COMPLEX ENGINEERING: Writing structural code, complex SQL optimizations, frontend/backend integrations, or API routing designs. "
        "Do NOT use for simple scripts, basic HTML/CSS fixes, or explaining basic programming concepts. "
        "Examples: 'Write an optimized PostgreSQL recursive CTE query to extract a hierarchical org chart with 50,000 nodes' or 'Design an asynchronous FastAPI routing layer handling WebSockets with JWT token authentication.'",
        
    "creative": 
        "HIGH COMPLEXITY CREATIVE: Nuanced editorial review, high-converting marketing copy, strict persona-driven brand interactions, or long-form narrative structure. "
        "Do NOT use for simple jokes, basic emails, casual brainstorming, or everyday text generation. "
        "Examples: 'Rewrite this landing page copy using a psychological persuasion framework tailored strictly for skeptical B2B CFOs' or 'Draft a multi-layered script with subtle foreshadowing for a corporate training video about insider security threats.'",
        
    "factual_long_context": 
        "MASSIVE CONTEXT: Large document corpus QA, legal compliance audits over multiple files, or analysis where the input text is massive (spanning >100k tokens). "
        "Examples: 'Extract every single liability clause from this 400-page merger agreement document' or 'Analyze our complete 10-K financial history over the last 5 years to find anomalies in R&D spending trends.'",
        
    "general_premium": 
        "MODERATE COMPLEXITY: Complex multi-turn chat dialogues, highly variable text instructions, or robust intent orchestration where nuance matters but deep domain expertise is not required. "
        "Examples: 'Acts as an empathetic customer support lead handling a complex refund escalations process' or 'Synthesize these 20 disparate user feedback reviews into 5 distinct product request categories.'",

    "agentic_professional":
        "PROFESSIONAL AUTOMATION WORKHORSE: Standardised business operations, multi-step backend microservice coordination, "
        "and intermediate agentic workflows executing rigid operational playbooks. "
        "Examples: 'Extract data from this standard invoice, map it to our internal JSON schema, verify the tax ID, and push it to the accounts payable endpoint.'",

    "agentic_fast": 
        "LATENCY-CRITICAL: Real-time interactions, high-speed function calling loops, or high-velocity data enrichment pipelines where speed is more important than deep reasoning. "
        "Examples: 'Power a real-time voice assistant conversational loop' or 'Provide low-latency function calling to fetch and return weather, stock ticker data, or user profile records inside a live chat UI.'",
        
    "general_budget": 
        "LOW COMPLEXITY EVERYDAY TASKS: Simple jokes, casual conversation, short email drafting, basic brainstorming, structural layout generation (JSON schema), and high-volume entity extraction. "
        "Use this for any standard creative or general request that is short and simple. "
        "Examples: 'Draft a quick 3-sentence thank you email to John' or 'Extract all names and email addresses from this unformatted block of text into a JSON array.'",
        
    "factual": 
        "BASIC KNOWLEDGE: Answering straightforward search/retrieval questions, summarization of short text blocks, exact question answering, or basic educational explanations. "
        "Examples: 'What is the capital of France?' or 'Explain the difference between synchronous and asynchronous code using a simple restaurant analogy.'",
        
    "fallback_cheapest": 
        "TRIVIAL/MICRO TASKS: Basic binary classification, filtering profanity/spam, single-turn string mutations, formatting text, or one-sentence generic queries requiring almost zero reasoning. "
        "Examples: 'Classify this text as Positive or Negative: [text]' or 'Strip out all special characters from this string and capitalize the first letter.'"
}


# prompt injection defense threshold (scale 1 to 5)
# any prompt scoring > 2.0 gets stopped immediately to protect our models
INJECTION_THRESHOLD = 2.0

# 5-level rubric used by TypeSafe SDK Score to rate prompt injection risk
INJECTION_RUBRIC_5_SCALE = [
    "Level 1: Benign and legitimate query with zero jailbreak or prompt injection intent.",
    "Level 2: Edge-case phrasing, technical discussion, or harmless curiosity without bypass intent.",
    "Level 3: Mild prompt injection, hypothetical roleplay attempting rule relaxation, or minor instruction override.",
    "Level 4: Direct prompt injection, DAN mode, instruction disregard ('ignore previous instructions'), or system prompt extraction attempt.",
    "Level 5: Severe malicious jailbreak exploit, base64/encoded payload injection, or hostile filter bypass.",
]

# regex patterns for offline fallback safety guard
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules|directions)",
    r"disregard\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)",
    r"forget\s+(all\s+)?(previous|prior)\s+(instructions|prompts|rules)",
    r"(bypass|disable|turn\s+off)\s+(all\s+)?(safety|guardrails|filters|restrictions)",
    r"(system\s+prompt|system\s+instructions)\s+(leak|reveal|show|dump|print|output)",
    r"(dan\s+mode|jailbreak|do\s+anything\s+now|developer\s+mode)",
]

