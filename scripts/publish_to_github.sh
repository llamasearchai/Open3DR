#!/bin/bash

# Open3DR GitHub Publishing Script
# Author: Nik Jois <nikjois@llamasearch.ai>

set -e

echo "Open3DR - Publishing to GitHub"
echo "==============================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Set git configuration
echo "Setting up git configuration..."
git config user.name "Nik Jois"
git config user.email "nikjois@llamasearch.ai"

# Add all files
echo "Adding all files to git..."
git add .

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Medical Data (HIPAA Compliance)
patient_data/
medical_data/
clinical_trials_data/
*.dcm
*.nii
*.nii.gz

# GPU/CUDA
*.ptx
*.cubin
*.fatbin

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
.tmp/

# Docker
.dockerignore

# Jupyter Notebooks
.ipynb_checkpoints

# Model weights and data
models/weights/*.pth
models/weights/*.ckpt
data/raw/
data/processed/

# Configuration files with secrets
config/secrets/
.env.local
.env.production
EOF
fi

# Commit changes
echo "Creating initial commit..."
git add .gitignore
git commit -m "Initial commit - Open3DR Medical AI Platform

- Advanced medical imaging and 3D reconstruction
- AI-powered diagnostics and clinical decision support
- Drug discovery and molecular modeling tools
- HIPAA-compliant medical data processing
- Real-time patient monitoring and collaboration
- Neural radiance fields for medical visualization
- Comprehensive test suite and benchmarking
- Docker containerization and cloud deployment

Author: Nik Jois <nikjois@llamasearch.ai>
Organization: LlamaSearch AI"

# Set up remote repository
echo "Setting up remote repository..."
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/llamasearchai/Open3DR.git

# Create and switch to main branch
git branch -M main

echo ""
echo "Repository prepared for publishing!"
echo ""
echo "Next steps:"
echo "1. Create the repository at: https://github.com/llamasearchai/Open3DR"
echo "2. Run: git push -u origin main"
echo ""
echo "Repository URL: https://github.com/llamasearchai/Open3DR"
echo "Author: Nik Jois <nikjois@llamasearch.ai>"
echo ""
echo "Note: Make sure you have created the repository on GitHub first!" 