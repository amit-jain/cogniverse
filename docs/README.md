# Documentation Structure

This directory contains detailed documentation for various components of the Cogniverse system.

## Directory Structure

### `/setup`
- **detailed_setup.md** - Comprehensive setup guide (formerly GETTING_STARTED.md)
  - Detailed prerequisites
  - Step-by-step installation
  - Configuration walkthrough
  - Troubleshooting common issues

### `/modal`
- **deploy_modal_vlm.md** - Step-by-step guide for deploying the Modal VLM service
  - Prerequisites and setup
  - Deployment instructions
  - Configuration guide
  - Troubleshooting tips
  - Cost estimation and scaling
- **setup_modal_vlm.py** - Automated setup script (optional helper)
  - Automatically installs Modal CLI
  - Deploys the service
  - Updates config.json
  - Tests the integration

### `/pipeline` (future)
- Detailed pipeline architecture documentation
- Pipeline configuration guide
- Custom pipeline step development

### `/testing`
- **README.md** - Testing overview and quick commands
- **search_client_testing.md** - How to test Vespa search client
- **vespa_search_strategies.md** - Complete guide to all 13 ranking strategies

## Quick Links

For getting started quickly, see the root directory docs:
- **QUICKSTART.md** - 3-step quick setup (recommended for first-time users)
- **Readme.md** - Complete system overview and architecture

For developer instructions and codebase guidelines:
- **CLAUDE.md** - AI assistant instructions and system architecture