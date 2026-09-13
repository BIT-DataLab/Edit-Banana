> !! DO NOT COMMIT THIS FILE !!

# T1.0-arrow-connection · Phase 1.0

> Enable users to automatically associate arrows with target shapes during image conversion

## Context

- **Dependency**: None (Phase 0 independent)
- **Boundary**: Backend arrow detection logic + optional manual override UI

## Current State

The system detects arrows during image processing but they exist as standalone shapes without connections to other elements. Users must manually connect arrows to shapes in DrawIO after conversion.

## Target State

Arrows are automatically associated with their source and target shapes based on:
1. Spatial proximity (arrow endpoints near shape boundaries)
2. Directional alignment (arrow pointing toward/away from shape)
3. Visual relationship (arrow color matching shape context)

## Tasks

### 1. Analyze existing arrow detection

- [x] Review current arrow detection implementation in `modules/arrow_processor.py`
- [x] Understand shape data structure and how arrows are represented
- [x] Identify existing arrow metadata (position, direction, endpoints)

### 2. Design arrow-shape association algorithm

- [x] Define spatial proximity threshold (configurable, default 50px)
- [x] Implement endpoint-to-shape-boundary distance calculation
- [x] Add directional vector analysis for arrow targeting
- [x] Create confidence scoring for potential connections
- [x] Handle ambiguous cases (multiple nearby shapes) - prefers strict containment

### 3. Implement association logic

- [x] Create `ArrowConnector` class in `modules/arrow_connector.py`
- [x] Add `process()` method that operates on ProcessingContext
- [x] Store connection metadata in element.metadata['arrow_connection']
- [x] Support manual override via connection metadata editing

### 4. Update data models

- [x] Added `metadata` field to ElementInfo dataclass for storing connection info
- [x] Arrow metadata includes: source_id, target_id, source_confidence, target_confidence

### 5. Create manual override UI (optional enhancement)

- [ ] Add connection editor component (deferred to future phase)
- [ ] Allow users to confirm/reject auto-connections (deferred)
- [ ] Enable manual connection creation (deferred)

### 6. Write tests

- [x] Unit tests: `tests/test_arrow_connector.py` (16 tests, all passing)
  - Single arrow → single shape
  - Multiple arrows → one shape
  - Arrow between two shapes
  - Ambiguous connection handling
  - Confidence threshold edge cases
- [x] Integration: Pipeline in `main.py` step [5] calls ArrowConnector
- [x] XML generation handles connected arrows with source/target attributes

## Done When

- [x] All Tasks checkbox checked (except deferred UI items)
- [x] `pytest tests/test_arrow_connector.py -v` passes (16 tests)
- [x] `python -m modules.arrow_connector` runs without errors
- [x] Pipeline integration complete in `main.py`
- [x] XML fragments generated with proper source/target for connected arrows

## Test Plan

**Manual verification**:
1. Upload diagram with arrows pointing to boxes
2. Verify output JSON includes `source_shape_id` and `target_shape_id` for arrows
3. Import into DrawIO and verify arrows maintain connections

**Edge cases**:
- Floating arrows (no nearby shapes) → no connection, log warning
- Multiple arrows near one shape → connect all within threshold
- Arrow with ambiguous targets → connect to highest confidence, flag for review
- Arrow endpoint inside shape → prefers containing shape over boundary proximity

## Implementation Notes

### ArrowConnector Algorithm
1. Separate arrows from shapes in the element list
2. For each arrow, get start and end points from `arrow_start`/`arrow_end` metadata
3. Calculate distance from each endpoint to all shape boundaries
4. Points strictly inside a shape return negative distance (preferred over boundary)
5. Sort candidates by distance and pick nearest within threshold
6. Calculate confidence score based on distance (exponential decay)
7. Store connection metadata in element.metadata['arrow_connection']

### Distance Calculation
- Strictly inside bbox: returns negative distance to nearest edge
- On boundary: returns 0
- Outside: returns Euclidean distance to nearest edge

### XML Generation
Connected arrows generate mxCell with:
- `edge="1"` attribute
- `source="{source_id}"` and/or `target="{target_id}"` attributes
- `mxPoint` elements for sourcePoint and targetPoint
