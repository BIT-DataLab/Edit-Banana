> !! DO NOT COMMIT THIS FILE !!

# T0-merge-stabilize · Phase 0

> Merge foundational PRs and resolve conflicts to establish a stable main branch

## Context

- **Dependency**: None (Phase 0, first task)
- **Boundary**: Git operations only, no code changes beyond merge conflict resolution

## Current State

11 open PRs with conflicts:
- **Duplicates**: #39 duplicates #36, #40 duplicates #38
- **Conflicts**: #38 vs #40 (CI/CD), #41 vs #42 (arrow_processor.py)

## Merge Order (Dependency Graph)

```
#25 (config-loader) → #31 (text-extraction) → #38 (CI/CD)
                                    ↓
                              #36 (Next.js frontend)
                                    ↓
                    #35 (arrow), #34 (SAM3), #37 (layer-merge)
                                    ↓
                         #41, #42 (arrow_processor)
```

## Tasks

### 1. Merge foundational PRs

- [ ] Merge PR #25 (config-loader improvements)
- [ ] Merge PR #31 (text-extraction fix)
- [ ] Merge PR #38 (CI/CD workflow) - close #40 as duplicate
- **File**: Git operations
- **验收**: `git log --oneline` shows clean merge commits
- **测试**: CI passes on merged branch

### 2. Merge frontend base

- [ ] Merge PR #36 (Next.js frontend) - close #39 as duplicate
- **File**: Git operations
- **验收**: Frontend code present in apps/web/
- **测试**: `npm run build` succeeds

### 3. Merge feature PRs

- [ ] Merge PR #35 (arrow improvements)
- [ ] Merge PR #34 (SAM3 integration)
- [ ] Merge PR #37 (layer-based XML merging)
- **File**: Git operations
- **验收**: All feature code integrated
- **测试**: pytest tests/core/ passes

### 4. Resolve arrow_processor conflicts

- [ ] Analyze #41 vs #42 differences
- [ ] Choose best implementation or merge both
- [ ] Resolve any merge conflicts
- **File**: `core/processors/arrow_processor.py`
- **验收**: No duplicate code, all tests pass
- **测试**: Arrow processing functional

## Done When

- [ ] All foundational PRs merged
- [ ] Duplicates (#39, #40) closed
- [ ] No merge conflicts remaining
- [ ] CI passes on main
- [ ] PR created with clean merge

## Test Plan

**Manual verification**:
1. Check `git log --graph` shows clean history
2. Verify no duplicate files
3. Run full test suite
4. Confirm CI passes
