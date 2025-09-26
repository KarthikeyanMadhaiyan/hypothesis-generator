hypothesis_schema = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "hypothesis": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {
            "type": "object",
            "properties": {
                "evidence": {"type": "array", "items": {"type": "string"}},
                "chain_of_thought": {"type": "array", "items": {"type": "string"}},
                "metadata_analysis": {
                    "type": "object",
                    "properties": {
                        "temporal_correlation": {"type": "number"},
                        "service_correlation": {"type": "number"},
                        "regional_correlation": {"type": "number"},
                        "similarity_score": {"type": "number"}
                    }
                },
                "llm_reasoning": {"type": "string"},
                "risk_factors": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["evidence", "chain_of_thought", "metadata_analysis", "llm_reasoning"]
        },
        "category": {"type": "string", "enum": ["infrastructure", "application", "network", "database", "external", "unknown"]},
        "priority": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
        "estimated_impact": {"type": "string"},
        "recommended_actions": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["id", "hypothesis", "confidence", "reasoning", "category", "priority"]
}

# Enhanced input schema for validation
input_schema = {
    "type": "object",
    "properties": {
        "anomaly": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string"},
                "timestamp": {"type": "string"},
                "metadata": {"type": "object"}
            },
            "required": ["id", "type", "timestamp", "metadata"]
        },
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"}
                },
                "required": ["id", "type"]
            }
        }
    },
    "required": ["anomaly", "candidates"]
}