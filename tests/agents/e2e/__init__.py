"""
End-to-end integration tests for multi-agent system.

These tests use real LLMs, actual DSPy optimization, and real backend services
to test complete workflows without mocks.

Test Categories:
- Real LLM Integration: Tests with actual Ollama models
- Real DSPy Optimization: Tests actual prompt optimization pipeline  
- Real Multi-Agent Workflow: Tests complete agent communication
- Performance Comparison: Tests optimized vs default agent performance

Requirements:
- Ollama server running locally with smollm3:8b model
- Optional: Vespa backend for video search integration
- Optional: Phoenix telemetry server for advanced metrics
"""