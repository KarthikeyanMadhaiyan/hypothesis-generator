import json
import random
from typing import List, Dict
from jsonschema import validate
from schema import hypothesis_schema

def generate_hypotheses(anomaly: Dict, candidates: List[Dict]) -> List[Dict]:
    hypotheses = []
    for idx, candidate in enumerate(candidates[:3]):  # ensure at least 3
        confidence = round(random.uniform(0.6, 0.95), 2)  # mock confidence
        reasoning = {
            "evidence": [
                f"Candidate type: {candidate['type']}",
                f"Service: {candidate.get('service', 'N/A')}"
            ],
            "chain_of_thought": [
                "Step 1: Match anomaly type with candidate metadata",
                "Step 2: Assess similarity or correlation",
                "Step 3: Assign confidence based on match strength"
            ]
        }
        hypotheses.append({
            "id": f"H{idx+1}",
            "hypothesis": f"{candidate['type']} may have caused {anomaly['type']}",
            "confidence": confidence,
            "reasoning": reasoning
        })
    return hypotheses


if __name__ == "__main__":
    with open("input.json", "r") as f:
        data = json.load(f)

    anomaly = data["anomaly"]
    candidates = data["candidates"]

    results = generate_hypotheses(anomaly, candidates)

    # schema validation
    for h in results:
        validate(instance=h, schema=hypothesis_schema)

    with open("output.json", "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Generated hypotheses saved to output.json")