// @ts-nocheck
import { test, expect, ElectronApplication, Page } from "@playwright/test";
import { launchApp, waitForBackendHealth } from "./helpers";

let app: ElectronApplication;
let page: Page;

test.beforeEach(async () => {
  app = await launchApp();
  page = await app.firstWindow();
  await waitForBackendHealth(page);
});

test.afterEach(async () => {
  await app.close();
});

test("backend health indicator shows OK", async () => {
  const dot = page.locator("#backend-dot");
  await expect(dot).toHaveClass(/ok/);
});

test("backend version is displayed", async () => {
  const ver = page.locator("#backend-version");
  const text = await ver.textContent();
  expect(text).toMatch(/^\d+\.\d+\.\d+$/);
});

test("convert and export buttons are disabled on startup", async () => {
  await expect(page.locator("#btn-convert")).toBeDisabled();
  await expect(page.locator("#btn-export")).toBeDisabled();
  await expect(page.locator("#btn-open-raw")).toBeEnabled();
});

test("error modal is hidden on startup", async () => {
  const modal = page.locator("#error-modal");
  const display = await modal.evaluate((el) => (el as HTMLElement).style.display);
  expect(display).toBe("none");
});

test("error modal shows with correct message", async () => {
  await page.evaluate((): void => {
    (window as unknown as Record<string, unknown>).__showTestErrorModal("Test error message");
  });
  const modal = page.locator("#error-modal");
  await expect(modal).toBeVisible();
  const msg = page.locator("#error-modal-message");
  await expect(msg).toContainText("Test error message");
});

test("dismiss button closes error modal", async () => {
  await page.evaluate(() => {
    (window as unknown as Record<string, unknown>).__showTestErrorModal("Test error");
  });
  const modal = page.locator("#error-modal");
  await expect(modal).toBeVisible();
  await page.locator("#btn-dismiss-error").click();
  await expect(modal).not.toBeVisible();
});

test("retry and dismiss buttons are both present in error modal", async () => {
  await page.evaluate(() => {
    (window as unknown as Record<string, unknown>).__showTestErrorModal("Test error");
  });
  await expect(page.locator("#btn-retry-conversion")).toBeVisible();
  await expect(page.locator("#btn-dismiss-error")).toBeVisible();
});

test("simulateError IPC enables error injection", async () => {
  await page.evaluate(() => {
    (window as unknown as Record<string, unknown>).electronAPI.backend.simulateError(true);
  });
  const result = await page.evaluate<string>(async () => {
    try {
      await (window as unknown as Record<string, unknown>).electronAPI.backend.convert({
        imageBuffer: new ArrayBuffer(0),
        imageName: "test.tiff",
        lutBuffers: [],
        lutNames: [],
      });
      return "success";
    } catch (err) {
      return (err as Error).message;
    }
  });
  expect(result).toContain("Simulated backend error");
  await page.evaluate(() => {
    (window as unknown as Record<string, unknown>).electronAPI.backend.simulateError(false);
  });
});

test("retry button triggers startConversion", async () => {
  await page.evaluate(() => {
    (window as unknown as Record<string, unknown>).electronAPI.backend.simulateError(true);
  });
  await page.evaluate(() => {
    (window as unknown as Record<string, unknown>).__showTestErrorModal("Simulated failure");
  });
  const modal = page.locator("#error-modal");
  await expect(modal).toBeVisible();
  await page.locator("#btn-retry-conversion").click();
  await expect(modal).toBeVisible();
});
