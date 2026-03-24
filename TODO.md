> !! Do not commit this file !!

# T0-cicd · Phase 0

> Every PR is automatically tested and deployed safely.

## Context

- **Dependency**: None (Phase 0)
- **Boundary**: Only add GitHub Actions workflows. No application code changes.

## Current Issues

- No automated testing on PRs
- No build verification
- Manual deployment process
- No quality gates

## Tasks

### 1. Create GitHub Actions Workflow

- [ ] Create `.github/workflows/ci.yml`
- [ ] Configure triggers: push to main, pull requests to main
- [ ] Add Node.js setup (18.x or 20.x)
- [ ] Add steps: checkout, install deps, lint, build

**File**: `.github/workflows/ci.yml` (create)
**Template**:
```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run lint
      - run: npm run build
```

### 2. Add Vercel Deployment

- [ ] Configure Vercel integration in GitHub
- [ ] Add preview deployments for PRs
- [ ] Add production deployment on merge to main

**Steps**:
1. Connect GitHub repo to Vercel
2. Configure environment variables if needed
3. Test deployment with this branch

### 3. Add PR Template

- [ ] Create `.github/pull_request_template.md`
- [ ] Include sections: Description, Test Plan, Screenshots

**File**: `.github/pull_request_template.md` (create)

### 4. Add Branch Protection

- [ ] Configure branch protection for main
- [ ] Require PR reviews
- [ ] Require status checks to pass
- [ ] Require up-to-date branches

**Note**: This requires admin access to repository settings

## Done When

- [ ] CI workflow runs on this PR
- [ ] Build passes in CI
- [ ] PR template appears when creating new PR
- [ ] Vercel deploys preview for this branch

---

### Testing Commands

```bash
# Push this branch to trigger CI
git push origin T0-cicd

# Verify workflow runs at:
# https://github.com/BIT-DataLab/Edit-Banana/actions
```
