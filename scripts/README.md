# Scripts Directory

Automation scripts for the Agentic Router system using the new `src/` structure.

## 📁 Scripts Overview

### 🚀 `run_optimization.py` - Complete Workflow
**Full end-to-end optimization and deployment pipeline**

```bash
# Run complete workflow (optimization + deployment + testing)
python scripts/run_optimization.py

# With custom config
python scripts/run_optimization.py --config my_config.json

# Skip specific steps
python scripts/run_optimization.py --skip-upload --skip-test
```

**What it does:**
1. ✅ Runs orchestrator optimization
2. ✅ Uploads artifacts to Modal volume
3. ✅ Deploys production API
4. ✅ Tests the deployed system

**Use when:** Starting from scratch or running full optimization

---

### 🏭 `deploy_production.py` - Production Only
**Deploy just the production API (after optimization is done)**

```bash
# Deploy production API using existing artifacts
python scripts/deploy_production.py
```

**What it does:**
1. ✅ Deploys production API to Modal
2. ✅ Tests API health and basic functionality
3. ✅ Shows usage examples

**Use when:** You have optimization artifacts and just want to deploy

---

### 🧪 `test_api.py` - Testing Only
**Comprehensive API testing and validation**

```bash
# Full test suite
python scripts/test_api.py

# Test specific API URL
python scripts/test_api.py --url https://your-api-url.modal.run

# Health check only
python scripts/test_api.py --health-only

# Quick tests
python scripts/test_api.py --quick
```

**What it does:**
1. ✅ Tests API health and model info
2. ✅ Runs comprehensive query tests
3. ✅ Validates NEW_PROPOSAL.md schema
4. ✅ Measures latency and accuracy

**Use when:** Testing deployed API or debugging issues

---

## 🔄 Common Workflows

### **First Time Setup:**
```bash
# 1. Configure models in config.json
# 2. Run complete optimization
python scripts/run_optimization.py
```

### **Optimization Options:**

#### Option 1: Just Optimization
```bash
# Run optimization only (no deployment)
python scripts/run_orchestrator.py

# Useful for testing optimization without deployment
```

#### Option 2: Full Workflow
```bash
# Run optimization + deployment + testing
python scripts/run_optimization.py

# Complete end-to-end workflow
```

### **Deploy After Optimization:**
```bash
# 1. Run orchestrator first
python scripts/run_orchestrator.py

# 2. Deploy production API with the artifacts
python scripts/deploy_production.py
```

### **Test Existing API:**
```bash
# Test the API by calling the endpoints directly
```

### **Development Cycle:**
```bash
# 1. Modify config
# 2. Test locally (new structure)
python scripts/run_orchestrator.py --test-models

# 3. Deploy and test
python scripts/deploy_production.py
```

### **New Scripts (src/ structure):**
```bash
# Run orchestrator only
python scripts/run_orchestrator.py [--config config.json] [--setup-only] [--test-models]

# Deploy Modal inference service manually
modal deploy src/inference/modal_inference_service.py
```

---

## 📋 Configuration

All scripts respect the main `config.json` file:

```yaml
# Model Configuration
teacher:
  model: "claude-3-5-sonnet-20241022"
  provider: "anthropic"
  
student:
  model: "google/gemma-3-1b-it"
  provider: "hosted"

# Optimization Settings
optimization:
  num_examples: 100
  num_candidates: 10
  num_trials: 20
```

---

## 🔍 Troubleshooting

### **Script Fails to Run:**
- Ensure you're in the `modal_inference/` directory
- Check Python path and dependencies
- Verify Modal CLI is installed and authenticated

### **Deployment Fails:**
- Check Modal authentication: `modal auth current`
- Verify Modal volume permissions
- Check API keys in environment/secrets

### **API Tests Fail:**
- Verify API URL is correct
- Check if artifacts were uploaded properly
- Test health endpoint first

### **Optimization Takes Too Long:**
- Reduce `num_examples` in config
- Use smaller teacher model
- Monitor Modal logs for issues

---

## 🛠️ Development

### **Adding New Scripts:**
1. Follow naming convention: `action_target.py`
2. Include comprehensive error handling
3. Add progress indicators
4. Update this README

### **Script Structure:**
```python
#!/usr/bin/env python3
"""
Script Description

What it does and when to use it.
"""

def main():
    """Main function with error handling."""
    try:
        # Implementation
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
```

---

## 📊 Output Examples

### **Successful Optimization:**
```
🚀 Starting Orchestrator Optimization...
✅ Orchestrator completed successfully
📤 Uploading artifacts to Modal volume...
✅ Artifacts uploaded successfully
🚀 Deploying Production API...
✅ Production API deployed successfully
🧪 Testing API...
✅ All tests passed

🎉 DEPLOYMENT COMPLETE!
🌐 API URL: https://agentic-router-production-route.modal.run
```

### **API Test Results:**
```
📋 Test 1/9: Video Tutorial Request
   ✅ Response: video/raw_results
   ⏱️ Latency: API 45ms, E2E 123ms
   ✅ Test passed

📊 TEST SUMMARY
✅ Passed: 9/9 tests
📈 Success Rate: 100.0%
⏱️ Latency: avg=52.3ms, min=31ms, max=89ms
```