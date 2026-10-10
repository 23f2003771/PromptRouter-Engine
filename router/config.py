import os
from dotenv import load_dotenv

# load environment variables from my .env file
load_dotenv()

# my API keys
TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")

# model mapped to each task type
MODEL_MAP = {
    # 1. Advanced Reasoning & Complex Coding (Chain-of-Thought)
    "reasoning_heavy": "deepseek-r1",             # Best specialized math/logic reasoning engine
    "reasoning_premium": "o3-reasoning",          # OpenAI's top-tier agentic reasoning model
    
    # 2. Flagship Multimodal & Production Workhorses
    "code": "claude-5-sonnet",                    # Gold standard for engineering, complex architecture
    "creative": "claude-5-sonnet",                # Best-in-class nuanced prose and system workflows
    "factual_long_context": "gemini-3.1-pro",     # Destroys 1M+ token RAG and massive document analysis
    "general_premium": "gpt-5.6-terra",           # Balanced flagship for robust standard API tasks

    # 3. High-Throughput & Cost-Optimized (Flash/Mini Tier)
    "agentic_fast": "gemini-3.8-flash",           # Blazing fast, low-latency agentic loops
    "general_budget": "gpt-5.6-luna",             # Fast OpenAI fallback for structured JSON extraction
    
    # 4. Rock-Bottom Baselines & Open Weights
    "factual": "deepseek-v4-flash",               # Incredible baseline knowledge per dollar
    "fallback_cheapest": "gemini-2.5-flash-lite"  # Absolute floor pricing default fallback
}

# estimated cost per 1k tokens (blended input/output for quick math)
# useful to track how much money I save by avoiding expensive models on simple tasks
MODEL_COSTS_PER_1K = {
    # Premium Reasoning Tier ($15 - $20 blended per 1M tokens)
    "o3-reasoning": 0.0250,        # Blended from $15.00 In / $60.00 Out
    "deepseek-r1": 0.0012,         # Highly disruptive CoT price (~$0.56 In / $1.68 Out)
    
    # Flagship Tier ($4 - $6 blended per 1M tokens)
    "claude-5-sonnet": 0.0040,     # Blended from $2.00 In / $10.00 Out
    "gpt-5.6-terra": 0.0045,       # Blended from $2.00 In / $12.00 Out
    "gemini-3.1-pro": 0.0045,      # Blended from $2.00 In / $12.00 Out (below 200k window)
    
    # Performance Flash/Mini Tier ($0.80 - $1.50 blended per 1M tokens)
    "gemini-3.8-flash": 0.0015,    # Blended from $0.75 In / $3.75 Out 
    "gpt-5.6-luna": 0.00045,       # Blended from $0.20 In / $1.20 Out
    
    # Extreme Budget & Open-Weights Tier (< $0.40 blended per 1M tokens)
    "deepseek-v4-flash": 0.00020,  # Blended from $0.14 In / $0.28 Out
    "gemini-2.5-flash-lite": 0.00018  # Blended from $0.10 In / $0.40 Out (Cheapest baseline)
}

# criteria I pass to TypeSafe SDK to classify incoming prompts
CLASSIFICATION_CRITERIA = {
"reasoning_heavy": 
        "HIGH COMPLEXITY MATH/LOGIC: Requires extended internal chain-of-thought. "
        "Use for multi-step math proofs, deep algorithmic architecture, execution-critical logic, or complex debugging. "
        "Do NOT use for basic coding or standard troubleshooting.",
        
    "reasoning_premium": 
        "HIGH-STAKES AGENTIC: Autonomous agent actions, multi-file codebase refactoring, or zero-shot complex system orchestration. "
        "Use when flawlessly navigating competing constraints is required.",
        
    "code": 
        "COMPLEX ENGINEERING: Writing structural code, complex SQL optimizations, frontend/backend integrations, or API routing designs. "
        "Do NOT use for simple scripts, basic HTML/CSS fixes, or explaining basic programming concepts.",
        
    "creative": 
        "HIGH COMPLEXITY CREATIVE: Nuanced editorial review, high-converting marketing copy, strict persona-driven brand interactions, or long-form narrative structure. "
        "Do NOT use for simple jokes, basic emails, casual brainstorming, or everyday text generation.",
        
    "factual_long_context": 
        "MASSIVE CONTEXT: Large document corpus QA, legal compliance audits over multiple files, or analysis where the input text is massive (spanning >100k tokens).",
        
    "general_premium": 
        "MODERATE COMPLEXITY: Complex multi-turn chat dialogues, highly variable text instructions, or robust intent orchestration where nuance matters but deep domain expertise is not required.",

    "agentic_fast": 
        "LATENCY-CRITICAL: Real-time interactions, high-speed function calling loops, or high-velocity data enrichment pipelines where speed is more important than deep reasoning.",
        
    "general_budget": 
        "LOW COMPLEXITY EVERYDAY TASKS: Simple jokes, casual conversation, short email drafting, basic brainstorming, structural layout generation (JSON schema), and high-volume entity extraction. "
        "Use this for any standard creative or general request that is short and simple.",
        
    "factual": 
        "BASIC KNOWLEDGE: Answering straightforward search/retrieval questions, summarization of short text blocks, exact question answering, or basic educational explanations.",
        
    "fallback_cheapest": 
        "TRIVIAL/MICRO TASKS: Basic binary classification, filtering profanity/spam, single-turn string mutations, formatting text, or one-sentence generic queries requiring almost zero reasoning."
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

