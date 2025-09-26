import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from enhanced_hypothesis_generator import EnhancedHypothesisGenerator
from schema import hypothesis_schema, input_schema
from jsonschema import validate


class TestEnhancedHypothesisGenerator:
    """Comprehensive test suite for the enhanced hypothesis generator."""
    
    @pytest.fixture
    def generator(self):
        """Create a generator instance with mocked dependencies."""
        with patch('enhanced_hypothesis_generator.openai.AsyncOpenAI'), \
             patch('enhanced_hypothesis_generator.SentenceTransformer'):
            return EnhancedHypothesisGenerator()
    
    @pytest.fixture
    def sample_anomaly(self):
        """Sample anomaly data."""
        return {
            "id": "A1",
            "type": "latency_spike",
            "timestamp": "2025-09-25T10:00:00Z",
            "metadata": {
                "region": "us-east-1",
                "service": "auth-api",
                "avg_latency": "800ms"
            }
        }
    
    @pytest.fixture
    def sample_candidate(self):
        """Sample candidate data."""
        return {
            "id": "C1",
            "type": "deployment",
            "service": "auth-api",
            "timestamp": "2025-09-25T09:50:00Z",
            "region": "us-east-1"
        }
    
    def test_temporal_correlation_calculation(self, generator, sample_anomaly, sample_candidate):
        """Test temporal correlation calculation."""
        correlation = generator._calculate_temporal_correlation(
            sample_anomaly['timestamp'],
            sample_candidate['timestamp']
        )
        assert 0 <= correlation <= 1
        assert correlation > 0.5  # Should be high for 10-minute difference
    
    def test_service_correlation_exact_match(self, generator, sample_anomaly, sample_candidate):
        """Test service correlation with exact match."""
        correlation = generator._calculate_service_correlation(
            sample_anomaly['metadata'],
            sample_candidate
        )
        assert correlation == 1.0
    
    def test_service_correlation_partial_match(self, generator):
        """Test service correlation with partial match."""
        anomaly_metadata = {"service": "auth-api"}
        candidate = {"service": "auth-service"}
        
        correlation = generator._calculate_service_correlation(anomaly_metadata, candidate)
        assert 0 < correlation < 1
    
    def test_service_correlation_no_match(self, generator):
        """Test service correlation with no match."""
        anomaly_metadata = {"service": "auth-api"}
        candidate = {"service": "payment-service"}
        
        correlation = generator._calculate_service_correlation(anomaly_metadata, candidate)
        assert correlation == 0.0
    
    def test_regional_correlation_match(self, generator, sample_anomaly, sample_candidate):
        """Test regional correlation with match."""
        correlation = generator._calculate_regional_correlation(
            sample_anomaly['metadata'],
            sample_candidate
        )
        assert correlation == 1.0
    
    def test_regional_correlation_no_match(self, generator):
        """Test regional correlation with no match."""
        anomaly_metadata = {"region": "us-east-1"}
        candidate = {"region": "us-west-2"}
        
        correlation = generator._calculate_regional_correlation(anomaly_metadata, candidate)
        assert correlation == 0.0
    
    def test_semantic_similarity_calculation(self, generator, sample_anomaly, sample_candidate):
        """Test semantic similarity calculation."""
        # Mock the embedding model
        generator.embedding_model = MagicMock()
        generator.embedding_model.encode.return_value = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
        
        similarity = generator._calculate_semantic_similarity(sample_anomaly, sample_candidate)
        assert 0 <= similarity <= 1
        generator.embedding_model.encode.assert_called_once()
    
    def test_metadata_analysis_integration(self, generator, sample_anomaly, sample_candidate):
        """Test comprehensive metadata analysis."""
        # Mock semantic similarity
        generator._calculate_semantic_similarity = MagicMock(return_value=0.8)
        
        analysis = generator._analyze_metadata_correlations(sample_anomaly, sample_candidate)
        
        assert "temporal_correlation" in analysis
        assert "service_correlation" in analysis
        assert "regional_correlation" in analysis
        assert "similarity_score" in analysis
        
        # Verify all values are in valid range
        for key, value in analysis.items():
            assert 0 <= value <= 1
    
    def test_confidence_score_calculation(self, generator):
        """Test confidence score calculation."""
        correlations = {
            "temporal_correlation": 0.8,
            "service_correlation": 1.0,
            "regional_correlation": 1.0,
            "similarity_score": 0.7
        }
        llm_confidence = 0.9
        
        confidence = generator._calculate_confidence_score(correlations, llm_confidence)
        assert 0.1 <= confidence <= 0.95  # Within configured bounds
    
    def test_categorize_hypothesis(self, generator):
        """Test hypothesis categorization."""
        assert generator._categorize_hypothesis("deployment", "latency") == "application"
        assert generator._categorize_hypothesis("db-connection", "error") == "database"
        assert generator._categorize_hypothesis("network-issue", "timeout") == "network"
        assert generator._categorize_hypothesis("unknown-type", "error") == "unknown"
    
    def test_determine_priority(self, generator):
        """Test priority determination."""
        assert generator._determine_priority(0.9, "outage") == "critical"
        assert generator._determine_priority(0.8, "latency_spike") == "high"
        assert generator._determine_priority(0.6, "warning") == "medium"
        assert generator._determine_priority(0.3, "info") == "low"
    
    @pytest.mark.asyncio
    async def test_llm_analysis_success(self, generator, sample_anomaly, sample_candidate):
        """Test successful LLM analysis."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "chain_of_thought": ["Step 1", "Step 2", "Step 3"],
            "evidence": ["Evidence 1", "Evidence 2"],
            "risk_factors": ["Risk 1", "Risk 2"],
            "detailed_reasoning": "Detailed analysis",
            "llm_confidence": 0.8
        })
        
        generator.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        correlations = {
            "temporal_correlation": 0.8,
            "service_correlation": 1.0,
            "regional_correlation": 1.0,
            "similarity_score": 0.7
        }
        
        result = await generator._get_llm_analysis(sample_anomaly, sample_candidate, correlations)
        
        reasoning, chain_of_thought, evidence, risk_factors, llm_confidence = result
        
        assert reasoning == "Detailed analysis"
        assert len(chain_of_thought) == 3
        assert len(evidence) == 2
        assert len(risk_factors) == 2
        assert llm_confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_llm_analysis_error_handling(self, generator, sample_anomaly, sample_candidate):
        """Test LLM analysis error handling."""
        generator.openai_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        
        correlations = {
            "temporal_correlation": 0.8,
            "service_correlation": 1.0,
            "regional_correlation": 1.0,
            "similarity_score": 0.7
        }
        
        result = await generator._get_llm_analysis(sample_anomaly, sample_candidate, correlations)
        
        reasoning, chain_of_thought, evidence, risk_factors, llm_confidence = result
        
        assert "Error in LLM analysis" in reasoning
        assert llm_confidence == 0.3  # Fallback confidence
    
    @pytest.mark.asyncio
    async def test_generate_recommendations(self, generator):
        """Test recommendation generation."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "recommendations": ["Check logs", "Review metrics", "Test connectivity"]
        })
        
        generator.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        recommendations = await generator._generate_recommendations(
            "deployment may have caused latency",
            "application",
            "high"
        )
        
        assert len(recommendations) == 3
        assert all(isinstance(rec, str) for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_generate_single_hypothesis(self, generator, sample_anomaly, sample_candidate):
        """Test single hypothesis generation."""
        # Mock all async methods
        generator._get_llm_analysis = AsyncMock(return_value=(
            "Detailed reasoning",
            ["Step 1", "Step 2"],
            ["Evidence 1", "Evidence 2"],
            ["Risk 1"],
            0.8
        ))
        generator._generate_recommendations = AsyncMock(return_value=["Action 1", "Action 2"])
        generator._calculate_semantic_similarity = MagicMock(return_value=0.7)
        
        hypothesis = await generator.generate_hypothesis(sample_anomaly, sample_candidate, "H01")
        
        # Validate against schema
        validate(instance=hypothesis, schema=hypothesis_schema)
        
        assert hypothesis["id"] == "H01"
        assert "confidence" in hypothesis
        assert 0 <= hypothesis["confidence"] <= 1
        assert hypothesis["category"] in ["infrastructure", "application", "network", "database", "external", "unknown"]
        assert hypothesis["priority"] in ["critical", "high", "medium", "low"]
    
    @pytest.mark.asyncio
    async def test_generate_multiple_hypotheses(self, generator, sample_anomaly):
        """Test multiple hypothesis generation."""
        candidates = [
            {"id": "C1", "type": "deployment", "service": "auth-api"},
            {"id": "C2", "type": "db-connection", "service": "user-db"},
            {"id": "C3", "type": "network-issue", "region": "us-east-1"}
        ]
        
        # Mock dependencies
        generator._get_llm_analysis = AsyncMock(return_value=(
            "Detailed reasoning",
            ["Step 1", "Step 2"],
            ["Evidence 1", "Evidence 2"],
            ["Risk 1"],
            0.8
        ))
        generator._generate_recommendations = AsyncMock(return_value=["Action 1"])
        generator._calculate_semantic_similarity = MagicMock(return_value=0.7)
        
        hypotheses = await generator.generate_hypotheses(sample_anomaly, candidates)
        
        assert len(hypotheses) == 3
        assert all("id" in h for h in hypotheses)
        assert all("confidence" in h for h in hypotheses)
        
        # Verify sorted by confidence
        confidences = [h["confidence"] for h in hypotheses]
        assert confidences == sorted(confidences, reverse=True)
    
    def test_input_schema_validation(self):
        """Test input schema validation."""
        valid_input = {
            "anomaly": {
                "id": "A1",
                "type": "latency_spike",
                "timestamp": "2025-09-25T10:00:00Z",
                "metadata": {"service": "auth-api"}
            },
            "candidates": [
                {"id": "C1", "type": "deployment"}
            ]
        }
        
        # Should not raise exception
        validate(instance=valid_input, schema=input_schema)
    
    def test_hypothesis_schema_validation(self):
        """Test hypothesis schema validation."""
        valid_hypothesis = {
            "id": "H01",
            "hypothesis": "deployment may have caused latency_spike",
            "confidence": 0.85,
            "reasoning": {
                "evidence": ["Evidence 1"],
                "chain_of_thought": ["Step 1"],
                "metadata_analysis": {
                    "temporal_correlation": 0.8,
                    "service_correlation": 1.0,
                    "regional_correlation": 1.0,
                    "similarity_score": 0.7
                },
                "llm_reasoning": "Detailed reasoning"
            },
            "category": "application",
            "priority": "high",
            "estimated_impact": "Significant impact",
            "recommended_actions": ["Action 1"]
        }
        
        # Should not raise exception
        validate(instance=valid_hypothesis, schema=hypothesis_schema)


# Integration test
@pytest.mark.asyncio
async def test_full_integration_with_sample_data():
    """Test full integration with sample data file."""
    with patch('enhanced_hypothesis_generator.openai.AsyncOpenAI'), \
         patch('enhanced_hypothesis_generator.SentenceTransformer'):
        
        generator = EnhancedHypothesisGenerator()
        
        # Mock LLM response
        generator._get_llm_analysis = AsyncMock(return_value=(
            "Analysis based on temporal proximity and service correlation",
            [
                "Deployment occurred 10 minutes before anomaly",
                "Same service affected",
                "Timeline suggests causal relationship"
            ],
            ["Deployment to auth-api", "Timing correlation"],
            ["Code changes", "Configuration issues"],
            0.8
        ))
        generator._generate_recommendations = AsyncMock(return_value=[
            "Review deployment logs",
            "Check for new errors",
            "Consider rollback if issues persist"
        ])
        generator._calculate_semantic_similarity = MagicMock(return_value=0.8)
        
        # Load sample data
        with open("input.json", "r") as f:
            data = json.load(f)
        
        # Validate input
        validate(instance=data, schema=input_schema)
        
        # Generate hypotheses
        results = await generator.generate_hypotheses(data["anomaly"], data["candidates"])
        
        # Validate results
        assert len(results) >= 3  # At least 3 hypotheses as required
        
        for hypothesis in results:
            validate(instance=hypothesis, schema=hypothesis_schema)
            assert 0 <= hypothesis["confidence"] <= 1
            assert hypothesis["category"] in ["infrastructure", "application", "network", "database", "external", "unknown"]
            assert hypothesis["priority"] in ["critical", "high", "medium", "low"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
