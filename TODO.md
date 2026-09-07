> !! DO NOT COMMIT THIS FILE !!

# T0-cicd-setup · Phase 0

> Setup CI/CD pipeline for automated testing and deployment

## Context

- **Dependency**: None (Phase 0 independent, but ideally after frontend merge)
- **Boundary**: Focus on GitHub Actions workflow

## Current State

- No CI/CD pipeline exists
- Manual testing and deployment only
- No automated quality gates

## Target State

- GitHub Actions workflow for CI/CD
- Automated testing on PR and push
- Build verification for frontend and backend
- Deployment-ready artifact generation

## Tasks

### 1. Create GitHub Actions workflow

- [ ] Create `.github/workflows/ci.yml`
- [ ] Configure Python setup (3.10+)
- [ ] Configure Node.js setup (18+)
- [ ] Add backend test execution (pytest)
- [ ] Add frontend build verification
- [ ] Add lint/type check steps
- **File**: `.github/workflows/ci.yml` (new)
- **验收**: Workflow runs on push and PR
- **测试**: Trigger workflow and verify all steps pass

### 2. Configure test execution

- [ ] Ensure pytest tests run in CI
- [ ] Skip tests that require external dependencies (SAM3, Tesseract)
- [ ] Configure test markers for unit vs integration tests
- [ ] Add test coverage reporting
- **File**: `pyproject.toml` or `pytest.ini` (modify)
- **验收**: `pytest tests/core/ -v` passes in CI
- **测试**: CI shows green test results

### 3. Add build artifacts

- [ ] Configure frontend build caching (npm)
- [ ] Configure Python dependency caching (pip)
- [ ] Generate build artifacts for deployment
- [ ] Add artifact upload step
- **File**: `.github/workflows/ci.yml` (modify)
- **验收**: Build artifacts available for download
- **测试**: Verify artifacts contain built files

### 4. Optional: Add deployment job

- [ ] Add staging deployment job (on develop branch)
- [ ] Add production deployment job (on main branch, with approval)
- [ ] Configure environment-specific variables
- **File**: `.github/workflows/ci.yml` (modify)
- **验收**: Deployment triggers on correct branches
- **测试**: Test deployment to staging

## Done When

- [ ] All Tasks checkbox checked
- [ ] CI workflow runs on push to main/develop
- [ ] CI workflow runs on PR creation/update
- [ ] All tests pass in CI environment
- [ ] Build artifacts generated successfully
- [ ] PR created and ready for merge

## Workflow Template

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/core/ -v

  build-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
      - name: Install dependencies
        run: cd apps/web && npm ci
      - name: Build
        run: cd apps/web && npm run build
```

## Test Plan

**Manual verification**:
1. Push workflow file to branch
2. Create test PR
3. Verify CI runs and passes
4. Check build artifacts are generated
