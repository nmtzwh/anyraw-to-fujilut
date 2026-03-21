# E2E Tests — FujiLUT Electron App

## Overview

Playwright-based end-to-end tests for the FujiLUT Electron application.

## Test coverage

- Backend health indicator on startup (shows OK status dot)
- Backend version display
- UI button state machine (convert/export disabled until inputs selected)
- Error modal: display on error, dismiss button, retry button
- Error modal: simulateError IPC channel injects deterministic backend errors

## Run

```bash
cd electron-wave2a

# First time: install dependencies
npm ci
npx playwright install --with-deps chromium

# Build the app (required before running tests)
npm run build

# Run tests headlessly
npm run test:e2e

# Run with headed browser (shows Electron window)
npm run test:e2e:headed

# Run with Playwright UI (interactive test runner)
npm run test:e2e:ui
```

## Architecture

- `helpers.ts` — `launchApp()` using `_electron` fixture, `waitForBackendHealth()` to poll for the green status dot
- `app.spec.ts` — main test suite; each test launches a fresh Electron app via `test.beforeEach`/`test.afterEach`

### Key test utilities

- `backend:simulateError(true)` IPC channel — injects a deterministic error into the next `backend:convert` call. Used to test error modal UX without needing a real broken backend.
- `window.__showTestErrorModal(msg)` — test hook that directly shows the error modal with an arbitrary message. Used for testing modal visibility and button wiring.

## CI

Playwright tests run in the `e2e-tests` job in `.github/workflows/build.yml` on Ubuntu and Windows. The job depends on `lint-and-test` (backend unit tests must pass first).
