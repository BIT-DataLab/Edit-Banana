> !! DO NOT COMMIT THIS FILE !!

# T1.0-error-recovery · Phase 1.0

> Provide clear error messages, retry logic, and partial results on conversion failures

## Context

- **Dependency**: None (Phase 0 independent)
- **Boundary**: Error handling across backend and frontend

## Current State

Errors during conversion result in:
- Generic error messages ("Conversion failed")
- No retry mechanism
- Complete loss of work on failure
- Silent failures in some edge cases

## Target State

- Structured error types with user-friendly messages
- Automatic retry with exponential backoff
- Partial results preserved and displayed
- Clear error boundaries with recovery options

## Tasks

### 1. Audit current error handling

- [ ] Review all try/except blocks in `app/services/converter.py`
- [ ] Identify silent failures and bare except clauses
- [ ] Map error sources: SAM3, OCR, LLM, file I/O, validation

### 2. Design exception hierarchy

- [x] Create `EditBananaException` base class
- [x] Define specific exceptions:
  - [x] `SegmentationError` (SAM3 failures)
  - [x] `OCRParsingError` (text extraction failures)
  - [x] `LLMProcessingError` (VLM API failures)
  - [x] `FileValidationError` (invalid input)
  - [x] `TimeoutError` (processing timeout)
  - [x] `XMLGenerationError` (XML generation failures)
  - [x] `ArrowProcessingError` (arrow detection failures)
  - [x] `ProcessingPartialResultError` (partial results)
- [x] Add error severity: `CRITICAL`, `RECOVERABLE`, `WARNING`
- [x] Implement auto-generated error codes
- [x] Add retry_allowed logic based on severity

### 3. Implement retry decorator

- [x] Create `modules/core/retry.py` with `@retry()` decorator
- [x] Support sync and async functions
- [x] Configurable: max_retries, base_delay, max_delay
- [x] Backoff strategies: fixed, linear, exponential
- [x] Exception filtering: exceptions_to_retry
- [x] Custom retry predicate: should_retry
- [x] Retry callback: on_retry
- [x] RetryContext for manual retry control
- [x] Global retry statistics: get_retry_stats, reset_retry_stats
- [x] Create `retry_with_defaults` convenience decorator

### 4. Update backend error handling

- [ ] Replace bare except clauses with specific handlers
- [ ] Wrap service calls with try/except + logging
- [ ] Ensure all errors propagate with context

### 5. Create partial results handling

- [x] Design partial result data structure (`PartialResultState` dataclass)
- [x] Implement `PartialResultsHandler` for save/load operations
- [x] Create `save_partial_results()` and `load_partial_results()` convenience functions
- [x] Generate partial DrawIO XML from saved state
- [x] Add summary method for debugging/information

### 6. Frontend error UI

- [ ] Create `ErrorBoundary` component with retry button
- [ ] Build `ErrorToast` with severity-based styling
- [ ] Add error detail expander (for debugging)
- [ ] Implement inline retry for failed conversions

### 7. Frontend error state management

- [ ] Extend conversion hook with error metadata
- [ ] Add retry counter and max retry state
- [ ] Store partial results in history (if available)

### 8. Write tests

- [x] Unit tests: Exception classes (`tests/core/test_exceptions.py` - 21 tests)
- [x] Unit tests: Retry logic (`tests/core/test_retry.py` - 20 tests)
- [x] Unit tests: Partial results (`tests/core/test_partial_results.py` - 19 tests)
- [ ] Component tests: Error UI rendering
- [ ] Integration tests: End-to-end error scenarios

## Done When

- [ ] All Tasks checkbox checked
- [x] `pytest tests/core/test_exceptions.py -v` passes (21 tests)
- [x] `pytest tests/core/test_retry.py -v` passes (20 tests)
- [x] `pytest tests/core/test_partial_results.py -v` passes (19 tests)
- [x] `pytest tests/core/ -v` passes (60 tests total)
- [ ] Manual test: Trigger error → see clear message → retry succeeds
- [ ] No bare except clauses remaining
- [ ] No lint/type errors

## Test Plan

**Manual verification**:
1. Upload corrupted image → see validation error with details
2. Disconnect network during conversion → auto-retry with backoff → user sees "Retrying..."
3. Max retries exceeded → see error with "Try again" button
4. Partial success → see extracted shapes even if text OCR failed

**Error scenarios to test**:
- Invalid file format
- File too large
- SAM3 segmentation timeout
- LLM API rate limit
- Network interruption mid-conversion
