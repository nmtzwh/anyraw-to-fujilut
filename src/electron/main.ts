/**
 * main.ts — Electron main process entry point.
 *
 * Responsibilities:
 * - Spawn Python FastAPI backend on port 19876.
 * - Create the BrowserWindow with secure defaults (contextIsolation, no nodeIntegration).
 * - Register IPC handlers (ipc.ts) and delegate network/file access to main.
 * - Shut down the Python subprocess cleanly on quit.
 * - Log all lifecycle events for debugging.
 */

import { app, BrowserWindow, dialog } from "electron";
import path from "node:path";
import { spawn, ChildProcess } from "node:child_process";
import { registerIpcHandlers } from "./ipc.js";

const BACKEND_PORT = 19876;
const BACKEND_HOST = "127.0.0.1";
const BACKEND_READY_TIMEOUT_MS = 30_000;

let mainWindow: BrowserWindow | null = null;
let backendProcess: ChildProcess | null = null;
let isQuitting = false;

/** Path to the Python backend entry point.
 *  In development (repo root) this is `backend/main.py`.
 *  In packaged Electron, this resolves to `resources/backend/main.py`.
 */
function resolveBackendScript(): string {
  // Check if running as a packaged app (electron is frozen)
  if (app.isPackaged) {
    return path.join((process as NodeJS.Process & { resourcesPath?: string }).resourcesPath ?? "", "backend", "main.py");
  }
  // Development: backend/ is a sibling of the electron build output
  // build/electron/main.js → parent → backend/main.py
  const devRoot = path.resolve(__dirname, "..", "..");
  return path.join(devRoot, "backend", "main.py");
}

/** Find a usable Python executable. */
function resolvePython(): string {
  for (const name of ["python3", "python"]) {
    try {
      const result = require("child_process").spawnSync(name, ["--version"], {
        stdio: "ignore",
      });
      if (result.status === 0) return name;
    } catch {
      // try next
    }
  }
  return "python3";
}

/** Start the Python FastAPI backend subprocess. */
function spawnBackend(): ChildProcess {
  const python = resolvePython();
  const script = resolveBackendScript();
  console.log(`[main] Spawning backend: ${python} ${script}`);
  const proc = spawn(python, [script], {
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, PORT: String(BACKEND_PORT) },
  });

  proc.stdout?.on("data", (chunk: Buffer) => {
    process.stdout.write(`[backend] ${chunk.toString().trimEnd()}\n`);
  });
  proc.stderr?.on("data", (chunk: Buffer) => {
    process.stderr.write(`[backend:err] ${chunk.toString().trimEnd()}\n`);
  });

  proc.on("error", (err) => {
    console.error("[main] Backend spawn error:", err);
  });

  proc.on("exit", (code, signal) => {
    console.log(`[main] Backend exited — code=${code} signal=${signal}`);
    if (mainWindow && !isQuitting) {
      dialog.showErrorBox(
        "Backend crashed",
        `The Python backend exited unexpectedly (code=${code}).\n` +
          "Click OK to quit the application."
      );
      app.quit();
    }
  });

  return proc;
}

/** Poll /health until the backend responds, then resolve. */
async function waitForBackendHealth(timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const resp = await fetch(`http://${BACKEND_HOST}:${BACKEND_PORT}/health`);
      if (resp.ok) {
        const body = await resp.json() as { status: string; version: string };
        console.log(`[main] Backend healthy: ${body.status} v${body.version}`);
        return;
      }
    } catch {
      // not ready yet
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`Backend did not become healthy within ${timeoutMs}ms`);
}

function createWindow(): BrowserWindow {
  console.log("[main] Creating BrowserWindow");
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    title: "FujiLUT",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  // In dev, load the local public/index.html
  if (!app.isPackaged) {
    const indexPath = path.resolve(__dirname, "..", "..", "public", "index.html");
    win.loadFile(indexPath);
  } else {
    // Packaged: load from resources/public/
    win.loadFile(path.join((process as NodeJS.Process & { resourcesPath?: string }).resourcesPath ?? "", "public", "index.html"));
  }

  win.webContents.on("did-fail-load", (_ev: Event, errorCode: number, errorDescription: string) => {
    console.error(`[main] Failed to load: ${errorCode} — ${errorDescription}`);
  });

  win.on("closed", () => {
    console.log("[main] BrowserWindow closed");
    mainWindow = null;
  });

  return win;
}

/** Global error guard — prevent silent crashes. */
process.on("uncaughtException", (err) => {
  console.error("[main] Uncaught exception:", err);
  dialog.showErrorBox("Uncaught exception", err.message);
});

process.on("unhandledRejection", (reason) => {
  console.error("[main] Unhandled rejection:", reason);
});

// ── App lifecycle ─────────────────────────────────────────────────────────────

app.whenReady().then(async () => {
  console.log(`[main] App ready — version ${app.getVersion()}`);

  // Register IPC handlers before creating the window
  registerIpcHandlers();
  console.log("[main] IPC handlers registered");

  // Spawn backend and wait for health
  backendProcess = spawnBackend();
  try {
    console.log(`[main] Waiting for backend health on port ${BACKEND_PORT}…`);
    await waitForBackendHealth(BACKEND_READY_TIMEOUT_MS);
    console.log("[main] Backend ready — creating window");
  } catch (err) {
    console.error("[main] Backend failed to start:", err);
    dialog.showErrorBox(
      "Backend failed",
      `Could not connect to the Python backend on port ${BACKEND_PORT}.\n` +
        `Error: ${err instanceof Error ? err.message : String(err)}\n\n` +
        "Make sure backend/requirements.txt is installed and try again."
    );
    app.quit();
    return;
  }

  mainWindow = createWindow();
});

// Quit when all windows are closed (standard desktop pattern)
app.on("window-all-closed", () => {
  console.log("[main] All windows closed");
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  // macOS: re-create window if dock icon clicked with no windows
  if (BrowserWindow.getAllWindows().length === 0) {
    mainWindow = createWindow();
  }
});

app.on("before-quit", () => {
  console.log("[main] before-quit — shutting down backend");
  isQuitting = true;
  if (backendProcess) {
    backendProcess.kill("SIGTERM");
    // Give it 3 s to exit gracefully
    setTimeout(() => {
      if (!backendProcess?.killed) {
        backendProcess?.kill("SIGKILL");
      }
    }, 3000);
  }
});

app.on("quit", () => {
  console.log("[main] App quit");
});
