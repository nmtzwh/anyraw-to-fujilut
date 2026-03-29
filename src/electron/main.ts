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
  if (app.isPackaged) {
    // extraResources are located in process.resourcesPath
    return path.join(process.resourcesPath, "backend", "main.py");
  }
  // Development: backend/ is a sibling of the electron build output
  // build/electron/main.js → parent → backend/main.py
  const devRoot = path.resolve(__dirname, "..", "..");
  return path.join(devRoot, "backend", "main.py");
}

function resolvePython(): string {
  if (app.isPackaged) {
    // Use the bundled venv python.exe
    return path.join(process.resourcesPath, "python-venv", "Scripts", "python.exe");
  }
  for (const name of ["python3", "python"]) {
    try {
      const result = require("node:child_process").spawnSync(name, ["--version"], {
        stdio: "ignore",
      });
      if (result.status === 0) return name;
    } catch {
      // try next
    }
  }
  return "python";
}

/** Start the Python FastAPI backend subprocess. */
function spawnBackend(): ChildProcess {
  const python = resolvePython();
  const script = resolveBackendScript();
  const cwd = app.isPackaged ? process.resourcesPath : path.resolve(__dirname, "..", "..");
  
  // Use quoted paths for Windows compatibility with spaces
  const command = `"${python}" "${script}" --host 127.0.0.1 --port ${BACKEND_PORT}`;
  console.log(`[main] Spawning backend with exec: ${command} in ${cwd}`);
  
  const proc = require("node:child_process").exec(command, {
    cwd,
    windowsHide: true,
    env: { 
      ...process.env, 
      PORT: String(BACKEND_PORT),
      PYTHONPATH: cwd
    },
  });

  proc.stdout?.on("data", (chunk: string) => {
    process.stdout.write(`[backend] ${chunk}\n`);
  });
  proc.stderr?.on("data", (chunk: string) => {
    process.stderr.write(`[backend:err] ${chunk}\n`);
  });

  proc.on("error", (err: any) => {
    console.error("[main] Backend exec error:", err);
  });

  proc.on("exit", (code: any, signal: any) => {
    console.log(`[main] Backend exited — code=${code} signal=${signal}`);
  });

  return proc;
}

/** Poll /health until the backend responds, then resolve. */
async function waitForBackendHealth(timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const resp = await fetch(`http://${BACKEND_HOST}:${BACKEND_PORT}/health`);
      if (resp.ok) return;
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
    // In packaged app, __dirname is dist/win-unpacked/resources/app/build/electron/
    // public/ is in dist/win-unpacked/resources/app/public/
    const indexPath = path.join(app.getAppPath(), "public", "index.html");
    win.loadFile(indexPath);
  }

  win.webContents.on("did-fail-load", (_ev: any, errorCode: number, errorDescription: string) => {
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

app.whenReady().then(() => {
  registerIpcHandlers();
  mainWindow = createWindow();

  // Spawn backend and wait for health
  backendProcess = spawnBackend();
  waitForBackendHealth(BACKEND_READY_TIMEOUT_MS).catch((err) => {
    console.error("[main] Backend failed to start:", err);
    if (!isQuitting) {
      dialog.showErrorBox("Backend error", `Could not connect to the Python backend.\n${err.message}`);
    }
  });
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
