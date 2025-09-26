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
                "chain_of_thought": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["evidence", "chain_of_thought"]
        }
    },
    "required": ["id", "hypothesis", "confidence", "reasoning"]
}