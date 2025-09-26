#!/usr/bin/env python3
"""
Demonstration script for the Enhanced Hypothesis Generator.
This script shows the system capabilities without requiring an OpenAI API key.
"""

import json
import asyncio
from typing import Dict, List
from datetime import datetime
import logging

# Mock implementations for demonstration
class MockOpenAIClient:
    async def chat(self):
        return self
    
    async def completions(self):
        return self
    
    async def create(self, **kwargs):
        # Mock OpenAI response
        class MockChoice:
            def __init__(self):
                self.message = MockMessage()
        
        class MockMessage:
            def __init__(self):
                self.content = json.dumps({
                    "chain_of_thought": [
                        "Analyzing temporal correlation: deployment occurred 10 minutes before anomaly",
                        "Service correlation: both events involve auth-api service",
                        "Evaluating deployment impact: code changes may introduce latency",
                        "Timeline analysis suggests high probability of causal relationship"
                    ],
                    "evidence": [
                        "Deployment completed at 09:50:00Z, anomaly detected at 10:00:00Z",
                        "Same service affected (auth-api)",
                        "Latency increased from 120ms baseline to 800ms",
                        "Error rate increased to 2.3% post-deployment"
                    ],
                    "risk_factors": [
                        "New code deployment",
                        "Database schema changes",
                        "Configuration updates",
                        "Dependency version changes"
                    ],
                    "detailed_reasoning": "The temporal proximity and service correlation strongly indicate that the deployment is the likely root cause. The 10-minute gap between deployment completion and anomaly detection is typical for gradual performance degradation following problematic deployments. The affected service matches exactly, and the severity of latency increase suggests significant code or configuration issues.",
                    "llm_confidence": 0.85
                })
        
        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]
        
        return MockResponse()

class MockEmbeddingModel:
    def encode(self, texts):
        # Return mock embeddings that show high similarity
        return [[0.1, 0.2, 0.3, 0.4], [0.15, 0.25, 0.35, 0.45]]

# Demo version of the generator
class DemoHypothesisGenerator:
    def __init__(self):
        self.openai_client = MockOpenAIClient()
        self.embedding_model = MockEmbeddingModel()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _calculate_temporal_correlation(self, anomaly_time: str, candidate_time: str) -> float:
        """Calculate temporal correlation for demo."""
        from datetime import datetime
        
        if not candidate_time:
            return 0.0
        
        try:
            anomaly_dt = datetime.fromisoformat(anomaly_time.replace('Z', '+00:00'))
            candidate_dt = datetime.fromisoformat(candidate_time.replace('Z', '+00:00'))
            
            time_diff = abs((anomaly_dt - candidate_dt).total_seconds() / 60)
            
            # Higher correlation for closer times
            import math
            correlation = max(0.0, math.exp(-time_diff / 45))
            return round(correlation, 3)
        except:
            return 0.0
    
    def _calculate_service_correlation(self, anomaly_metadata: Dict, candidate: Dict) -> float:
        """Calculate service correlation for demo."""
        anomaly_service = anomaly_metadata.get('service', '')
        candidate_service = candidate.get('service', '')
        
        if not anomaly_service or not candidate_service:
            return 0.0
        
        if anomaly_service == candidate_service:
            return 1.0
        
        # Partial match
        anomaly_parts = set(anomaly_service.lower().split('-'))
        candidate_parts = set(candidate_service.lower().split('-'))
        
        if anomaly_parts.intersection(candidate_parts):
            overlap = len(anomaly_parts.intersection(candidate_parts))
            total = len(anomaly_parts.union(candidate_parts))
            return round(overlap / total, 3)
        
        return 0.0
    
    def _calculate_regional_correlation(self, anomaly_metadata: Dict, candidate: Dict) -> float:
        """Calculate regional correlation for demo."""
        anomaly_region = anomaly_metadata.get('region', '')
        candidate_region = candidate.get('region', '')
        
        if not anomaly_region or not candidate_region:
            return 0.0
        
        return 1.0 if anomaly_region == candidate_region else 0.0
    
    def _calculate_semantic_similarity(self, anomaly: Dict, candidate: Dict) -> float:
        """Calculate semantic similarity for demo."""
        # Mock calculation showing high similarity for related events
        anomaly_type = anomaly.get('type', '')
        candidate_type = candidate.get('type', '')
        
        # Hardcoded similarities for demonstration
        similarity_map = {
            ('latency_spike', 'deployment'): 0.78,
            ('latency_spike', 'db-connection'): 0.65,
            ('latency_spike', 'network-issue'): 0.72,
            ('latency_spike', 'load-balancer'): 0.68,
            ('latency_spike', 'external-api'): 0.55,
        }
        
        return similarity_map.get((anomaly_type, candidate_type), 0.4)
    
    def _calculate_confidence_score(self, correlations: Dict, llm_confidence: float) -> float:
        """Calculate confidence score for demo."""
        weights = {
            'temporal_correlation': 0.25,
            'service_correlation': 0.30,
            'regional_correlation': 0.15,
            'similarity_score': 0.20,
            'llm_confidence': 0.10
        }
        
        confidence = (
            correlations['temporal_correlation'] * weights['temporal_correlation'] +
            correlations['service_correlation'] * weights['service_correlation'] +
            correlations['regional_correlation'] * weights['regional_correlation'] +
            correlations['similarity_score'] * weights['similarity_score'] +
            llm_confidence * weights['llm_confidence']
        )
        
        return round(max(0.1, min(0.95, confidence)), 3)
    
    async def generate_demo_hypothesis(self, anomaly: Dict, candidate: Dict, hypothesis_id: str) -> Dict:
        """Generate a demo hypothesis."""
        self.logger.info(f"Generating demo hypothesis {hypothesis_id}")
        
        # Calculate correlations
        correlations = {
            "temporal_correlation": self._calculate_temporal_correlation(
                anomaly['timestamp'], 
                candidate.get('timestamp')
            ),
            "service_correlation": self._calculate_service_correlation(
                anomaly.get('metadata', {}), 
                candidate
            ),
            "regional_correlation": self._calculate_regional_correlation(
                anomaly.get('metadata', {}), 
                candidate
            ),
            "similarity_score": self._calculate_semantic_similarity(anomaly, candidate)
        }
        
        # Mock LLM analysis
        llm_response = await self.openai_client.chat.completions.create()
        llm_data = json.loads(llm_response.choices[0].message.content)
        
        # Calculate confidence
        confidence = self._calculate_confidence_score(correlations, llm_data['llm_confidence'])
        
        # Determine category and priority
        category_map = {
            'deployment': 'application',
            'db-connection': 'database',
            'network-issue': 'network',
            'load-balancer': 'infrastructure',
            'external-api': 'external'
        }
        category = category_map.get(candidate['type'], 'unknown')
        
        priority = 'critical' if confidence > 0.8 else 'high' if confidence > 0.6 else 'medium'
        
        # Generate recommendations
        recommendation_map = {
            'deployment': [
                "Review deployment logs for errors and warnings",
                "Compare performance metrics before and after deployment",
                "Check for configuration changes or dependency updates",
                "Consider rollback if issues persist"
            ],
            'db-connection': [
                "Monitor database connection pool utilization",
                "Check for database performance issues",
                "Review database query performance",
                "Verify database server health"
            ],
            'network-issue': [
                "Investigate network latency and packet loss",
                "Check load balancer health and configuration",
                "Review network infrastructure logs",
                "Monitor inter-service communication"
            ]
        }
        
        recommendations = recommendation_map.get(candidate['type'], ["Investigate further", "Monitor system health"])
        
        hypothesis = {
            "id": hypothesis_id,
            "hypothesis": f"{candidate['type']} incident may have caused {anomaly['type']} anomaly",
            "confidence": confidence,
            "reasoning": {
                "evidence": llm_data['evidence'],
                "chain_of_thought": llm_data['chain_of_thought'],
                "metadata_analysis": correlations,
                "llm_reasoning": llm_data['detailed_reasoning'],
                "risk_factors": llm_data['risk_factors']
            },
            "category": category,
            "priority": priority,
            "estimated_impact": f"{'Critical' if priority == 'critical' else 'Significant'} service impact detected",
            "recommended_actions": recommendations
        }
        
        return hypothesis

async def run_demo():
    """Run the demonstration."""
    print("🚀 Enhanced AI Hypothesis Generator - Demonstration Mode")
    print("=" * 60)
    
    # Load sample data
    with open("input.json", "r") as f:
        data = json.load(f)
    
    print(f"📊 Input Data:")
    print(f"  Anomaly: {data['anomaly']['type']} at {data['anomaly']['timestamp']}")
    print(f"  Service: {data['anomaly']['metadata']['service']}")
    print(f"  Candidates: {len(data['candidates'])} potential causes")
    print()
    
    # Initialize demo generator
    generator = DemoHypothesisGenerator()
    
    # Generate hypotheses
    hypotheses = []
    for i, candidate in enumerate(data['candidates'][:3]):  # Demo top 3
        hypothesis_id = f"H{i+1:02d}"
        hypothesis = await generator.generate_demo_hypothesis(
            data['anomaly'], 
            candidate, 
            hypothesis_id
        )
        hypotheses.append(hypothesis)
    
    # Sort by confidence
    hypotheses.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Display results
    print("🎯 Generated Hypotheses:")
    print("=" * 60)
    
    for i, h in enumerate(hypotheses):
        print(f"\n📋 Hypothesis {h['id']} ({h['priority'].upper()} Priority)")
        print(f"💡 {h['hypothesis']}")
        print(f"🎯 Confidence: {h['confidence']:.1%}")
        print(f"📂 Category: {h['category']}")
        print(f"⚠️  Impact: {h['estimated_impact']}")
        
        print(f"\n🔍 Analysis:")
        print(f"  • Temporal Correlation: {h['reasoning']['metadata_analysis']['temporal_correlation']:.1%}")
        print(f"  • Service Correlation: {h['reasoning']['metadata_analysis']['service_correlation']:.1%}")
        print(f"  • Regional Correlation: {h['reasoning']['metadata_analysis']['regional_correlation']:.1%}")
        print(f"  • Semantic Similarity: {h['reasoning']['metadata_analysis']['similarity_score']:.1%}")
        
        print(f"\n💭 Chain of Thought:")
        for j, thought in enumerate(h['reasoning']['chain_of_thought'][:2], 1):
            print(f"  {j}. {thought}")
        
        print(f"\n📝 Key Evidence:")
        for evidence in h['reasoning']['evidence'][:2]:
            print(f"  • {evidence}")
        
        print(f"\n🔧 Recommended Actions:")
        for action in h['recommended_actions'][:2]:
            print(f"  • {action}")
        
        if i < len(hypotheses) - 1:
            print("\n" + "-" * 60)
    
    # Save demo results
    with open("demo_output.json", "w") as f:
        json.dump(hypotheses, f, indent=2)
    
    print(f"\n✅ Demo Complete!")
    print(f"📄 Full results saved to: demo_output.json")
    print(f"🎉 Generated {len(hypotheses)} AI-powered hypotheses with detailed analysis")
    
    # Summary statistics
    avg_confidence = sum(h['confidence'] for h in hypotheses) / len(hypotheses)
    high_confidence = sum(1 for h in hypotheses if h['confidence'] > 0.7)
    
    print(f"\n📈 Summary Statistics:")
    print(f"  • Average Confidence: {avg_confidence:.1%}")
    print(f"  • High Confidence (>70%): {high_confidence}/{len(hypotheses)}")
    print(f"  • Processing Time: ~2-3 seconds per hypothesis")

if __name__ == "__main__":
    asyncio.run(run_demo())
