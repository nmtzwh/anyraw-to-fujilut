import { _electron, ElectronApplication, Page } from "@playwright/test";

export async function waitForBackendHealth(page: Page, timeout = 20_000): Promise<void> {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const dot = await page.$("#backend-dot");
    const cls = dot ? await dot.getAttribute("class") : null;
    if (cls?.includes("ok")) return;
    await page.waitForTimeout(500);
  }
  throw new Error("Backend did not become ready within timeout");
}

export async function launchApp(): Promise<ElectronApplication> {
  const app = await _electron.launch({
    args: [process.cwd()],
    env: { ...process.env, NODE_ENV: "test" },
  });
  return app;
}
