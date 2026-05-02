> !! DO NOT COMMIT THIS FILE !!

# T0-preview · Phase 0

> Implement preview generation with watermark, blur, and rate limiting to increase conversion

## Context

- **Dependency**: None (can be developed in parallel with existing API)
- **Boundary**: Focus on preview MVP; payment integration out of scope for now

## Preview MVP Specification

### Requirements

1. **Low-res preview (50% quality)** with diagonal "EditBanana Preview" watermark
2. **View-only display** (no download, right-click disabled)
3. **Auto-expire after 5 minutes** then auto-delete
4. **Blur partial - bottom 30%** blurred with "Pay $X to unlock full quality" CTA
5. **Rate limit** - max 3 previews per IP per hour

## Tasks

### Backend: Preview Generation Endpoint

- [ ] Create `/api/v1/jobs/{job_id}/preview` endpoint
- [ ] Generate low-res image (50% quality JPEG) from source
- [ ] Add diagonal "EditBanana Preview" watermark using PIL/Pillow
- [ ] Blur bottom 30% of image with blur overlay
- [ ] Store preview with 5-minute TTL in Redis/memory
- [ ] Implement IP-based rate limiting (3 previews/hour)
- [ ] Auto-cleanup expired previews via background task

**File**: `apps/api/routes/preview.py` (new)
**File**: `apps/api/services/preview_generator.py` (new)
**File**: `apps/api/middleware/rate_limit.py` (new)
**验收**:
- Given uploaded image, When preview requested, Then returns watermarked low-res image with bottom 30% blurred
- Given same IP, When 4th preview requested within 1 hour, Then returns 429 error
- Given preview generated, When 5 minutes elapsed, Then file is deleted

### Frontend: Preview Modal

- [ ] Create `PreviewModal` component with view-only display
- [ ] Disable right-click context menu on preview image
- [ ] Show blur overlay on bottom 30% with CTA button
- [ ] Display "Pay $X to unlock" CTA (UI only, no payment logic yet)
- [ ] Add countdown timer showing expiration
- [ ] Add close button and "Download Full" button (triggers payment flow later)

**File**: `apps/web/src/components/preview/preview-modal.tsx` (new)
**File**: `apps/web/src/app/sections/preview-section.tsx` (new)
**验收**:
- Given conversion complete, When preview button clicked, Then modal opens with preview
- Given preview open, When right-click attempted, Then context menu blocked
- Given preview open, When 5 minutes pass, Then modal auto-closes with expiration message

### API Integration

- [ ] Add `getPreviewUrl(jobId)` function in `api.ts`
- [ ] Add preview state management in upload section
- [ ] Integrate preview button in completion state

**File**: `apps/web/src/lib/api.ts` (modify)
**File**: `apps/web/src/app/sections/upload-section.tsx` (modify)
**验收**:
- Given job completed, When user clicks "Preview", Then preview modal opens with generated preview

## Testing Plan

1. **Unit Tests**: Preview generation with various image sizes
2. **Integration Tests**: Rate limiting, TTL cleanup
3. **Manual Tests**:
   - Upload → Convert → Preview flow
   - Rate limit enforcement
   - Auto-expiration

## Done When

- [ ] All tasks checkbox checked
- [ ] Preview endpoint returns watermarked, partially-blurred images
- [ ] Rate limiting working (3 previews/hour/IP)
- [ ] Frontend preview modal displays correctly
- [ ] Auto-cleanup removes expired previews
- [ ] Manual test: full upload→convert→preview flow works
- [ ] `/pr2default` created and merged
