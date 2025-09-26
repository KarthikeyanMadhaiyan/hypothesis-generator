import json
import logging
import asyncio
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from jsonschema import validate, ValidationError

from config import settings
from schema import hypothesis_schema, input_schema

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)

class EnhancedHypothesisGenerator:
    def __init__(self):
        """Initialize the enhanced hypothesis generator with LLM and embeddings."""
        self.openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        logger.info(f"Initialized with model: {settings.openai_model}")
        
    def _calculate_temporal_correlation(self, anomaly_time: str, candidate_time: Optional[str]) -> float:
        """Calculate temporal correlation between anomaly and candidate."""
        if not candidate_time:
            return 0.0
            
        try:
            anomaly_dt = datetime.fromisoformat(anomaly_time.replace('Z', '+00:00'))
            candidate_dt = datetime.fromisoformat(candidate_time.replace('Z', '+00:00'))
            
            # Calculate time difference in minutes
            time_diff = abs((anomaly_dt - candidate_dt).total_seconds() / 60)
            
            # Higher correlation for closer times (exponential decay)
            # Perfect correlation at 0 minutes, 50% at 30 minutes, ~0% after 2 hours
            correlation = max(0.0, np.exp(-time_diff / 45))
            return round(correlation, 3)
        except Exception as e:
            logger.warning(f"Error calculating temporal correlation: {e}")
            return 0.0
    
    def _calculate_service_correlation(self, anomaly_metadata: Dict, candidate: Dict) -> float:
        """Calculate service-level correlation."""
        anomaly_service = anomaly_metadata.get('service', '')
        candidate_service = candidate.get('service', '')
        
        if not anomaly_service or not candidate_service:
            return 0.0
            
        # Exact match
        if anomaly_service == candidate_service:
            return 1.0
            
        # Partial match (e.g., auth-api vs auth-service)
        anomaly_parts = set(anomaly_service.lower().split('-'))
        candidate_parts = set(candidate_service.lower().split('-'))
        
        if anomaly_parts.intersection(candidate_parts):
            overlap = len(anomaly_parts.intersection(candidate_parts))
            total = len(anomaly_parts.union(candidate_parts))
            return round(overlap / total, 3)
            
        return 0.0
    
    def _calculate_regional_correlation(self, anomaly_metadata: Dict, candidate: Dict) -> float:
        """Calculate regional/location correlation."""
        anomaly_region = anomaly_metadata.get('region', '')
        candidate_region = candidate.get('region', '')
        
        if not anomaly_region or not candidate_region:
            return 0.0
            
        return 1.0 if anomaly_region == candidate_region else 0.0
    
    def _calculate_semantic_similarity(self, anomaly: Dict, candidate: Dict) -> float:
        """Calculate semantic similarity using embeddings."""
        try:
            # Create text representations
            anomaly_text = f"{anomaly['type']} {json.dumps(anomaly.get('metadata', {}))}"
            candidate_text = f"{candidate['type']} {json.dumps(candidate)}"
            
            # Generate embeddings
            embeddings = self.embedding_model.encode([anomaly_text, candidate_text])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return round(float(similarity), 3)
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _analyze_metadata_correlations(self, anomaly: Dict, candidate: Dict) -> Dict:
        """Comprehensive metadata analysis."""
        return {
            "temporal_correlation": self._calculate_temporal_correlation(
                anomaly['timestamp'], 
                candidate.get('time', candidate.get('timestamp'))
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
    
    def _calculate_confidence_score(self, correlations: Dict, llm_confidence: float) -> float:
        """Calculate intelligent confidence score based on multiple factors."""
        # Weighted combination of different correlation factors
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
        
        # Ensure within bounds
        return round(max(settings.min_confidence, min(settings.max_confidence, confidence)), 3)
    
    def _categorize_hypothesis(self, candidate_type: str, anomaly_type: str) -> str:
        """Categorize hypothesis based on types."""
        type_mapping = {
            'deployment': 'application',
            'db-connection': 'database',
            'database': 'database',
            'network-issue': 'network',
            'network': 'network',
            'infrastructure': 'infrastructure',
            'server': 'infrastructure',
            'external': 'external',
            'api': 'application'
        }
        
        return type_mapping.get(candidate_type.lower(), 'unknown')
    
    def _determine_priority(self, confidence: float, anomaly_type: str) -> str:
        """Determine priority based on confidence and anomaly type."""
        critical_types = ['outage', 'security', 'data_loss']
        high_types = ['latency_spike', 'error_rate', 'performance']
        
        if any(ct in anomaly_type.lower() for ct in critical_types):
            return 'critical' if confidence > 0.7 else 'high'
        elif any(ht in anomaly_type.lower() for ht in high_types):
            return 'high' if confidence > 0.6 else 'medium'
        else:
            return 'medium' if confidence > 0.5 else 'low'
    
    async def _get_llm_analysis(self, anomaly: Dict, candidate: Dict, correlations: Dict) -> Tuple[str, List[str], str, List[str], float]:
        """Get detailed analysis from LLM."""
        prompt = f"""
        You are an expert in incident response and root cause analysis. Analyze the following anomaly and potential cause:

        ANOMALY:
        - Type: {anomaly['type']}
        - Timestamp: {anomaly['timestamp']}
        - Metadata: {json.dumps(anomaly.get('metadata', {}), indent=2)}

        POTENTIAL CAUSE:
        - Type: {candidate['type']}
        - Details: {json.dumps(candidate, indent=2)}

        CORRELATION ANALYSIS:
        - Temporal correlation: {correlations['temporal_correlation']}
        - Service correlation: {correlations['service_correlation']}
        - Regional correlation: {correlations['regional_correlation']}
        - Semantic similarity: {correlations['similarity_score']}

        Please provide:
        1. A detailed chain-of-thought reasoning (as JSON array of strings)
        2. Key evidence points (as JSON array of strings)
        3. Risk factors to consider (as JSON array of strings)
        4. A detailed explanation of how this cause could lead to the anomaly
        5. Your confidence in this hypothesis (0.0 to 1.0)

        Respond ONLY with valid JSON in this format:
        {{
            "chain_of_thought": ["step 1", "step 2", "step 3"],
            "evidence": ["evidence 1", "evidence 2"],
            "risk_factors": ["risk 1", "risk 2"],
            "detailed_reasoning": "detailed explanation here",
            "llm_confidence": 0.75
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return (
                result.get('detailed_reasoning', ''),
                result.get('chain_of_thought', []),
                result.get('evidence', []),
                result.get('risk_factors', []),
                float(result.get('llm_confidence', 0.5))
            )
        except Exception as e:
            logger.error(f"Error getting LLM analysis: {e}")
            return (
                f"Error in LLM analysis: {str(e)}",
                ["Unable to generate chain of thought due to LLM error"],
                [f"Candidate type: {candidate['type']}", f"Anomaly type: {anomaly['type']}"],
                ["LLM analysis unavailable"],
                0.3
            )
    
    async def _generate_recommendations(self, hypothesis: str, category: str, priority: str) -> List[str]:
        """Generate actionable recommendations."""
        prompt = f"""
        Given this hypothesis: "{hypothesis}"
        Category: {category}
        Priority: {priority}

        Generate 3-5 specific, actionable recommendations for investigation or mitigation.
        Respond with a JSON array of strings.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('recommendations', [])
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return [f"Review {category} systems", "Check recent changes", "Monitor for related issues"]
    
    async def generate_hypothesis(self, anomaly: Dict, candidate: Dict, hypothesis_id: str) -> Dict:
        """Generate a single comprehensive hypothesis."""
        logger.info(f"Generating hypothesis {hypothesis_id} for {candidate['type']} -> {anomaly['type']}")
        
        # Analyze metadata correlations
        correlations = self._analyze_metadata_correlations(anomaly, candidate)
        
        # Get LLM analysis
        llm_reasoning, chain_of_thought, evidence, risk_factors, llm_confidence = await self._get_llm_analysis(
            anomaly, candidate, correlations
        )
        
        # Calculate final confidence
        confidence = self._calculate_confidence_score(correlations, llm_confidence)
        
        # Determine category and priority
        category = self._categorize_hypothesis(candidate['type'], anomaly['type'])
        priority = self._determine_priority(confidence, anomaly['type'])
        
        # Generate hypothesis statement
        hypothesis = f"{candidate['type']} incident may have caused {anomaly['type']} anomaly"
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(hypothesis, category, priority)
        
        # Estimate impact
        impact_levels = {
            'critical': 'Severe service disruption, immediate action required',
            'high': 'Significant performance impact, urgent investigation needed',
            'medium': 'Moderate impact, should be investigated within hours',
            'low': 'Minor impact, can be investigated during business hours'
        }
        
        result = {
            "id": hypothesis_id,
            "hypothesis": hypothesis,
            "confidence": confidence,
            "reasoning": {
                "evidence": evidence,
                "chain_of_thought": chain_of_thought,
                "metadata_analysis": correlations,
                "llm_reasoning": llm_reasoning,
                "risk_factors": risk_factors
            },
            "category": category,
            "priority": priority,
            "estimated_impact": impact_levels[priority],
            "recommended_actions": recommendations
        }
        
        return result
    
    async def generate_hypotheses(self, anomaly: Dict, candidates: List[Dict]) -> List[Dict]:
        """Generate comprehensive hypotheses for all candidates."""
        logger.info(f"Generating hypotheses for {len(candidates)} candidates")
        
        # Sort candidates by potential relevance (using quick similarity check)
        scored_candidates = []
        for candidate in candidates:
            quick_score = self._calculate_semantic_similarity(anomaly, candidate)
            scored_candidates.append((candidate, quick_score))
        
        # Sort by score and take top candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored_candidates[:settings.max_hypotheses]
        
        # Generate hypotheses concurrently
        tasks = []
        for i, (candidate, _) in enumerate(top_candidates):
            hypothesis_id = f"H{i+1:02d}"
            task = self.generate_hypothesis(anomaly, candidate, hypothesis_id)
            tasks.append(task)
        
        hypotheses = await asyncio.gather(*tasks)
        
        # Sort by confidence
        hypotheses.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"Generated {len(hypotheses)} hypotheses")
        return hypotheses

async def main():
    """Main execution function."""
    try:
        # Validate input
        with open("input.json", "r") as f:
            data = json.load(f)
        
        validate(instance=data, schema=input_schema)
        logger.info("Input validation passed")
        
        # Initialize generator
        generator = EnhancedHypothesisGenerator()
        
        # Generate hypotheses
        results = await generator.generate_hypotheses(
            data["anomaly"], 
            data["candidates"]
        )
        
        # Validate each hypothesis
        for i, hypothesis in enumerate(results):
            try:
                validate(instance=hypothesis, schema=hypothesis_schema)
                logger.info(f"Hypothesis {i+1} validation passed")
            except ValidationError as e:
                logger.error(f"Hypothesis {i+1} validation failed: {e}")
                raise
        
        # Save results
        with open("output.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"✅ Generated {len(results)} validated hypotheses:")
        for h in results:
            print(f"  • {h['id']}: {h['hypothesis']} (confidence: {h['confidence']}, priority: {h['priority']})")
        
        logger.info("Enhanced hypothesis generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
