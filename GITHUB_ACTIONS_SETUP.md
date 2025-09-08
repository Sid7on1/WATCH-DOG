# 🚀 GitHub Actions Setup for WATCHDOG Systems

## 📋 Overview

Your WATCHDOG system now has **3 GitHub Actions workflows** that run automatically without user interaction:

### 🔄 **Hybrid Workflow** (Recommended)
- **File**: `.github/workflows/hybrid-watchdog.yml`
- **Schedule**: Runs daily at 2:00 AM UTC
- **Behavior**: Automatically alternates between original and enhanced modes
- **Cycle**: 1 day original → 2 days enhanced → repeat

### 📊 **Original Workflow**
- **File**: `.github/workflows/original-watchdog.yml`
- **Trigger**: Manual only (workflow_dispatch)
- **Behavior**: Runs your original arXiv-to-code system
- **Coverage**: All arXiv domains, general-purpose implementations

### 🚀 **Enhanced Workflow**
- **File**: `.github/workflows/enhanced-watchdog.yml`
- **Trigger**: Manual only (workflow_dispatch)
- **Behavior**: Runs enhanced AI/ML specialized system
- **Focus**: RAG, NLP, Agents, MLOps, Computer Vision

## 🔧 Required GitHub Secrets

Set these secrets in your GitHub repository settings (`Settings > Secrets and variables > Actions`):

### 🔑 **API Keys** (At least one required)
```
OPEN_API                 # OpenRouter API key
OPENROUTER_API_KEY       # Alternative name for OpenRouter
groq_API                 # Groq API key
cohere_API               # Cohere API key
GEMINI_API_KEY           # Google Gemini API key (optional)
```

### 🐙 **GitHub Integration**
```
GITHUB_PAT               # GitHub Personal Access Token
GITHUB_TOKEN             # Alternative (usually auto-provided)
API_GITHUB               # Alternative name for GitHub token
```

## 📊 **What Each System Does**

### 🔄 **Hybrid System** (Auto-runs daily)
```
Day 1: Original Mode
├── Scrapes papers from ALL arXiv domains
├── Creates general-purpose implementations
├── Stores artifacts in WATCHDOG_memory repo
└── Creates individual project repositories

Day 2-3: Enhanced Mode
├── Scrapes AI/ML focused papers (RAG, NLP, Agents, MLOps)
├── Creates specialized, production-ready implementations
├── Stores artifacts in WATCHDOG_memory repo
└── Creates individual project repositories

Day 4: Original Mode (cycle repeats)
```

### 📊 **Original System**
```
Manual Trigger Only:
├── Broad arXiv paper scraping
├── General relevance selection
├── Basic project templates
├── 4 general-purpose coding agents
├── 200-400 lines of code per file
├── Stores in WATCHDOG_memory + creates project repos
```

### 🚀 **Enhanced System**
```
Manual Trigger Only:
├── AI/ML focused paper scraping
├── Implementation potential analysis
├── Advanced project templates
├── 4 specialized coding agents:
│   ├── Agent 1: RAG & Vector Databases
│   ├── Agent 2: Advanced NLP & Transformers
│   ├── Agent 3: AI Agents & Multi-Agent Systems
│   └── Agent 4: MLOps/CI-CD & Computer Vision
├── 600-1200+ lines of production-ready code per file
├── Stores in WATCHDOG_memory + creates project repos
```

## 🗂️ **Repository Structure**

### 📁 **WATCHDOG_memory Repository**
- **Purpose**: Stores all artifacts, seen titles, metadata
- **Contents**: PDFs, text chunks, relevant papers, project plans
- **Persistence**: Maintains state between runs

### 🚀 **Individual Project Repositories**
- **Created automatically** for each generated project
- **Naming**: Based on project name (e.g., `enhanced_rag_system_project`)
- **Contents**: Complete project code, README, requirements.txt
- **Ready to use**: Can be cloned and run immediately

## ⏰ **Scheduling & Triggers**

### 🔄 **Automatic (Hybrid)**
```yaml
schedule:
  - cron: '0 2 * * *'  # Daily at 2:00 AM UTC
```

### 🔧 **Manual Triggers**
```bash
# Via GitHub UI: Actions tab > Select workflow > Run workflow

# Hybrid with specific mode
Inputs: mode = "original" or "enhanced"

# Original system
Inputs: papers_per_domain = "2"

# Enhanced system  
Inputs: papers_per_domain = "3"
```

## 📊 **Monitoring & Results**

### 📈 **Artifacts**
Each workflow uploads artifacts containing:
- Generated projects and code
- Summary reports
- System logs and status
- Configuration files

### 📋 **Summary Reports**
- **hybrid_summary.md**: Mode information, artifact counts, system status
- **original_summary.md**: Broad coverage results
- **enhanced_summary.md**: AI/ML specialized results, project types

### 🔍 **Logs**
- Real-time execution logs in GitHub Actions
- Component-by-component progress tracking
- Error reporting and debugging information

## 🛠️ **Configuration Options**

### 🔄 **Hybrid System Configuration**
The hybrid system automatically manages its schedule, but you can configure:

```bash
# In your repository, create/edit hybrid_mode_config.json:
{
  "auto_alternate": true,
  "original_papers_per_domain": 2,
  "enhanced_papers_per_domain": 3,
  "current_mode": "original",
  "cycle_day": 1
}
```

### ⚙️ **Environment Variables**
Set in workflow files or as repository secrets:
```yaml
SELECTOR_MIN_SCORE: '3.0'      # Minimum relevance score
WATCHDOG_FINALIZE: '1'         # Enable cleanup after completion
PYTHONUNBUFFERED: '1'          # Real-time log output
```

## 🚨 **Error Handling**

### 🔄 **Automatic Recovery**
- **API failures**: Automatic failover between API providers
- **Rate limits**: Intelligent backoff and retry
- **Component failures**: Graceful degradation, partial results saved

### 📊 **Failure Reporting**
- Detailed error logs in GitHub Actions
- Failure reports uploaded as artifacts
- System status preserved for debugging

## 🎯 **Best Practices**

### 🔑 **Security**
- ✅ All API keys stored as GitHub Secrets
- ✅ No sensitive data in logs or artifacts
- ✅ Automatic cleanup of temporary files

### 📊 **Resource Management**
- ✅ 4-6 hour timeouts prevent infinite runs
- ✅ Artifact retention limited to 30 days
- ✅ Automatic cleanup with `WATCHDOG_FINALIZE=1`

### 🔄 **Reliability**
- ✅ Multiple API providers for redundancy
- ✅ Graceful handling of component failures
- ✅ State persistence in WATCHDOG_memory repo

## 🚀 **Getting Started**

### 1. **Set up Secrets**
Go to your repository `Settings > Secrets and variables > Actions` and add:
- At least one API key (`OPEN_API`, `groq_API`, or `cohere_API`)
- GitHub token (`GITHUB_PAT`)

### 2. **Enable Workflows**
- The hybrid workflow will start running automatically daily
- Manual workflows can be triggered from the Actions tab

### 3. **Monitor Results**
- Check the Actions tab for execution logs
- Download artifacts to see generated projects
- Check your WATCHDOG_memory repository for persistent data
- Look for new project repositories created automatically

### 4. **Customize (Optional)**
- Adjust papers per domain in workflow inputs
- Modify schedules in workflow files
- Configure hybrid system behavior

## 📈 **Expected Results**

### 🔄 **Daily Hybrid Execution**
- **Day 1**: 2-5 general projects from broad arXiv coverage
- **Day 2**: 3-6 specialized AI/ML projects (RAG, NLP, etc.)
- **Day 3**: 3-6 more specialized AI/ML projects
- **Repeat cycle**

### 🗂️ **Repository Growth**
- **WATCHDOG_memory**: Grows with artifacts and metadata
- **Project repos**: New repositories created for each project
- **Code quality**: Enhanced projects have 600-1200+ lines per file

### 🎯 **Project Types**
- **Original**: Any research area, general implementations
- **Enhanced**: RAG systems, NLP models, AI agents, MLOps pipelines, CV systems

Your WATCHDOG system now runs completely autonomously, alternating between broad research coverage and specialized AI/ML project generation! 🚀