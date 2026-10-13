import vinext from "vinext";
import { readFileSync } from "node:fs";
import http from "node:http";
import { defineConfig, type ViteDevServer } from "vite";
import hostingConfig from "./.openai/hosting.json";
import { readExecutionProfile } from "./scripts/execution-profile.mjs";
import { sites } from "./build/sites-vite-plugin";

const SITE_CREATOR_PLACEHOLDER_DATABASE_ID =
  "00000000-0000-4000-8000-000000000000";

const { d1, r2 } = hostingConfig;

// macOS Seatbelt blocks FSEvents, so Codex previews need polling for HMR.
const isCodexSeatbeltSandbox = process.env.CODEX_SANDBOX === "seatbelt";
const managedLinux = readExecutionProfile() === "managed-linux";

const localBindingConfig = {
  main: "vinext/server/fetch-handler",
  compatibility_flags: ["nodejs_compat"],
  d1_databases: d1
    ? [
        {
          binding: d1,
          database_name: "site-creator-d1",
          database_id: SITE_CREATOR_PLACEHOLDER_DATABASE_ID,
        },
      ]
    : [],
  r2_buckets: r2
    ? [
        {
          binding: r2,
          bucket_name: "site-creator-r2",
        },
      ]
    : [],
};

export default defineConfig(async () => {
  // Use Miniflare's local Request.cf placeholder unless fetching is requested.
  process.env.CLOUDFLARE_CF_FETCH_ENABLED ??= "false";
  process.env.WRANGLER_SEND_METRICS ??= "false";

  // Keep Wrangler and Miniflare state project-local. These are non-secret tool
  // settings; application environment belongs in ignored `.env*` files.
  process.env.WRANGLER_WRITE_LOGS ??= "false";
  process.env.WRANGLER_LOG_PATH ??= ".wrangler/logs";
  process.env.WRANGLER_REGISTRY_PATH ??= ".wrangler/dev-registry";
  process.env.MINIFLARE_REGISTRY_PATH ??= ".wrangler/registry";

  // Wrangler snapshots its log path while the Cloudflare plugin is imported.
  const { cloudflare } = await import("@cloudflare/vite-plugin");

  return {
    server: {
      ...(managedLinux
        ? { host: "0.0.0.0", allowedHosts: ["terminal.local"] }
        : {}),
      ...(isCodexSeatbeltSandbox
        ? { watch: { useFsEvents: false, usePolling: true } }
        : {}),
    },
    plugins: [
      {
        name: "private-local-backend",
        configureServer(server: ViteDevServer) {
          server.middlewares.use("/backend", (req, res) => {
            const origin = req.headers.origin;
            const expected = `http://${req.headers.host}`;
            if (
              (origin && origin !== expected) ||
              req.headers["sec-fetch-site"] === "cross-site"
            ) {
              res.statusCode = 403;
              res.end("Forbidden");
              return;
            }
            let token: string;
            try {
              token = readFileSync(
                new URL("../backend/.data/admin.key", import.meta.url),
                "utf8",
              ).trim();
            } catch {
              res.statusCode = 503;
              res.end(
                JSON.stringify({ detail: "Start the local backend first" }),
              );
              return;
            }
            const headers = {
              ...req.headers,
              host: "127.0.0.1:8000",
              authorization: `Bearer ${token}`,
            };
            delete headers.origin;
            const upstream = http.request(
              {
                hostname: "127.0.0.1",
                port: 8000,
                path: req.url,
                method: req.method,
                headers,
              },
              (response) => {
                res.writeHead(response.statusCode || 502, response.headers);
                response.pipe(res);
              },
            );
            upstream.on("error", () => {
              if (!res.headersSent)
                res.writeHead(503, { "Content-Type": "application/json" });
              res.end(
                JSON.stringify({
                  detail:
                    "The local backend is offline. Start it and try again.",
                }),
              );
            });
            req.pipe(upstream);
          });
        },
      },
      vinext(),
      sites({ mockAuth: !managedLinux }),
      cloudflare({
        viteEnvironment: { name: "rsc", childEnvironments: ["ssr"] },
        inspectorPort: false,
        config: localBindingConfig,
      }),
    ],
  };
});
