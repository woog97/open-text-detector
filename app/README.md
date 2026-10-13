# vinext-starter

A clean full-stack starter running on [vinext](https://github.com/cloudflare/vinext), with optional Cloudflare D1 and Drizzle support.

## Prerequisites

- Node.js `>=22.13.0`
- Portable: Windows, macOS, or Linux; no Bash required
- Managed Linux: managed Linux runtime with Bash, `flock`, `curl`, `sha256sum`, and GNU `timeout`
- Git is required only for publishing

## Sites Lifecycle

The Sites initializer copies the shared starter and selects managed-linux only when `SITES_MANAGED_LINUX_CONTAINER=1`; otherwise it selects portable. It saves the selection only in ignored `.sites-runtime/execution-profile.json`. Both profiles copy/configure first, then use the plugin's separate `install-dependencies.mjs` step to measure installation independently. Edit source under `app/` and follow the Sites skill for installation, preview, builds, and publishing.

Run `node <plugin-root>/scripts/configure-execution-profile.mjs` only when the profile is unknown for the current checkout and environment. Profile changes do not alter tracked source or require reinstalling otherwise-valid dependencies; restart an existing preview to use the new selection. Do not commit or upload `.sites-runtime/`.

This starter does not use `wrangler.jsonc`.

`install:ci` runs `npm ci` once against the shared lockfile, disables parent-workspace discovery, and includes required dev/optional dependencies despite production/omit settings. Sharp defaults to prebuilt binaries unless explicitly configured otherwise. Do not overlap installers.

- **Portable:** Preserve host HOME, npm cache, registry, proxy, temporary paths, retry/concurrency settings, and lifecycle-script policy. Use `--prefer-offline --no-audit --no-fund`.
- **Managed Linux:** Use the existing project-local HOME/cache/tmp setup and Linux install lock, tarball preflight, and timeout. Restore the image-seeded npm cache only when its lockfile hash matches; retain network fallback. Builds keep their existing timeout. These helpers are not invoked by the portable profile.

`scripts/sites-env.mjs` preserves the caller's HOME, npm cache, proxy, XDG, and temporary-directory configuration while defaulting Wrangler and Miniflare state to the checkout. If npm reports an unwritable cache, select a writable path with `npm_config_cache` for that install. The `dev` and `start` scripts also keep Wrangler logs inside the checkout. Generated `.sites-runtime/` and `.wrangler/` directories are disposable and ignored by Git.

On portable, `npm run dev` uses `vinext dev` with HMR, starting at port 5173. Vinext records the running server in ignored `.vinext/` state, rejects an ordinary duplicate launch, and recovers stale state after a stopped process; exactly simultaneous starts can race. Pass `--port <port>` or `--hostname <host>` after `npm run dev --` when needed; keep portable previews on loopback.

For browser QA on managed Linux, use `sites-preview start`. The project's dev script runs Vite and accepts the supervisor's `--host 0.0.0.0 --port 4173 --strictPort` arguments. The internal browser uses `http://terminal.local:4173/`; it is not a user-facing URL. The supervisor owns the preview lifecycle. The ignored local profile survives the supervisor's cleared process environment.

The portable profile simulates ChatGPT sign-in only for loopback development requests. Visit `/signin-with-chatgpt?return_to=/` to sign in as `local_seedy` (`seedy@sites.test`, display name `Seedy`) and `/signout-with-chatgpt?return_to=/` to sign out. The development cookie preserves that identity across server restarts. Mock auth is disabled in the managed-linux profile and is not included in production builds; hosted authentication remains dispatch-owned.

The Worker uses `vinext/server/fetch-handler`, including Vinext's config-aware image handling. After building, `npm start` runs that Worker locally through Wrangler on `127.0.0.1`, sharing `.wrangler/state` with dev preview and local D1 migrations; it does not deploy the site or simulate sign-in. Use the URL printed by the server. Pass `npm start -- --port <port>` to select a different built-preview port.

Local previews use Miniflare's placeholder `Request.cf` metadata without a network lookup. Set `CLOUDFLARE_CF_FETCH_ENABLED=true` to opt into fetching preview metadata; this setting does not change hosted request metadata.

Local tool usage metrics are disabled by default. Set `WRANGLER_SEND_METRICS=true` to opt in.

## Included Shape

- edit site code under `app/`
- `app/chatgpt-auth.ts` provides optional dispatch-owned ChatGPT sign-in helpers
- `.openai/hosting.json` declares optional Sites D1 and R2 bindings
- `vite.config.ts` simulates declared bindings for local development
- `db/index.ts` reads the D1 binding from the Cloudflare Worker environment
- `db/schema.ts` starts intentionally empty
- `@cloudflare/workers-types` provides Worker types; `cloudflare-env.d.ts` declares optional `DB`/`BUCKET` bindings—update these declarations if binding names change
- `examples/d1/` contains an optional D1 example surface
- `drizzle.config.ts` supports local migration generation when needed

## Workspace Auth Headers

Signed-in visitors receive both `oai-authenticated-user-id` and `oai-authenticated-user-email`. Private Sites require every visitor to sign in; public Sites may also have anonymous visitors, for whom neither header is present.

The user ID is stable for the same user on the same Site and different across Sites. Use it as the durable user key; use email and name for display or contact purposes.

SIWC-authenticated workspace sites may also receive `oai-authenticated-user-full-name` when the user's SIWC profile has a non-empty `name` claim. The full-name value is percent-encoded UTF-8 and is accompanied by `oai-authenticated-user-full-name-encoding: percent-encoded-utf-8`.

Treat the full name as optional and fall back to email when it is absent:

```tsx
import { headers } from "next/headers";

export default async function Home() {
  const requestHeaders = await headers();
  const userId = requestHeaders.get("oai-authenticated-user-id");
  const email = requestHeaders.get("oai-authenticated-user-email");
  const encodedFullName = requestHeaders.get("oai-authenticated-user-full-name");
  const fullName =
    encodedFullName &&
    requestHeaders.get("oai-authenticated-user-full-name-encoding") ===
      "percent-encoded-utf-8"
      ? decodeURIComponent(encodedFullName)
      : null;

  const displayName = fullName ?? email;
  // ...
}
```

## Optional Dispatch-Owned ChatGPT Sign-In

Import the ready-to-use helpers from `app/chatgpt-auth.ts` when the site needs optional or required ChatGPT sign-in:

- Use `getChatGPTUser()` for optional signed-in UI.
- Use the returned `userId` as the stable user key for user-owned records; do not use email as a durable identifier.
- Use `requireChatGPTUser(returnTo)` for server-rendered pages that should send anonymous visitors through Sign in with ChatGPT.
- In a Server Component, start sign-in with `<a href={chatGPTSignInPath(returnTo)} target="_top">`. The auth helper module is server-only; do not import it into a Client Component.
- Do not use `fetch`, XHR, a client-side router, or a framework link that can prefetch the sign-in route. SIWC must start as a top-level navigation.
- Never request the AuthAPI authorization endpoint directly. The dispatch-owned `/signin-with-chatgpt` route must start the SIWC flow.
- Use `chatGPTSignOutPath(returnTo)` for browser sign-out links or actions.
- Pass a same-origin relative `returnTo` path for the destination after sign-in or sign-out. The helper validates and safely encodes it.
- Mark protected pages with `export const dynamic = "force-dynamic"` because they depend on per-request identity headers.

Dispatch owns `/signin-with-chatgpt`, `/signout-with-chatgpt`, `/callback`, the OAuth cookies, and identity header injection. Do not implement app routes for those reserved paths. Routes that do not import and call the helper remain anonymous-compatible.

SIWC establishes identity only; it does not prove workspace membership. Use the Sites hosting platform's access policy controls for workspace-wide restrictions, or enforce explicit server-side membership or allowlist checks.

Use SIWC for account pages, user-specific dashboards, saved records, and write actions tied to the current ChatGPT user. Leave public content anonymous.

## Local D1 migrations

For a D1-backed local preview, generate SQL with `npm run db:generate`. Build once through the Sites skill's build entrypoint (or `npm run build` for standalone use) to generate `dist/server/wrangler.json`, rebuilding if bindings change. From the project root, apply each pending migration in order:

```sh
node --import ./scripts/sites-env.mjs ./node_modules/wrangler/bin/wrangler.js d1 execute DB --local --config dist/server/wrangler.json --persist-to .wrangler/state --file drizzle/0000_example.sql
```

Replace the filename with the pending migration and `DB` with your D1 binding name if different. Use `.wrangler/state`, not `.wrangler/state/v3`; Wrangler adds the versioned directories. Do not replay migrations already applied locally. This updates only the preview database; publishing applies production migrations separately.

## Diagnostic Commands

- `npm run install:ci`: perform the one locked dependency install
- `npm run dev`: start the Vite/Vinext development server
- `npm run build`: build the deployable Sites artifact
- `npm run start`: preview the built Worker locally with D1/R2 support
- `npm run db:generate`: generate Drizzle migrations after schema changes

When using the Sites plugin, follow its skill instructions for installation, builds, and publishing. These npm commands remain available for standalone use.

The portable build runs Vinext directly without a host `timeout` command. The managed-linux build uses `scripts/build-verified.sh` and its existing `SITES_BUILD_TIMEOUT` setting.

## Learn More

- [vinext Documentation](https://github.com/cloudflare/vinext)
- [Drizzle D1 Guide](https://orm.drizzle.team/docs/get-started/d1-new)


## Fast paper browsing — October 3, 2026

Deployed version `fac8fb03-8f31-49d4-8541-6e9dc6572478` replaces the discovery page's full catalogue/metadata downloads with `/backend/v1/browse`. `scripts/publish_browse_snapshot.py` inventories verified R2 PDFs at publication time, builds immutable lean search and first-page snapshots, verifies uploads, then switches `indexes/browse/current.json`. The existing ICLR indexer calls it after index publication. Run `python3 scripts/publish_browse_snapshot.py` after any out-of-band paper-list update. Raw uploads outside that flow appear in the browse snapshot after its next publication.

The default first page streams a precomputed snapshot containing20 papers, collection counts, area counts and popular keywords, without scanning R2 or parsing the complete index. Search/filter/page requests return20 rows; selected-paper responses add reader navigation and related papers. Immutable revision-keyed edge caching is shared across visitors; browser/manifest freshness is15seconds. Saved-list requests bypass shared caching. The browser shows its last saved public preview immediately, refreshes in the background and checks again every60seconds while visible. It does not download the entire library in the background. Classification UI loads lazily and retains its existing list data path. Individual PDF/result/metadata routes now resolve published IDs from the results manifest or inspect only the requested public hash prefix, avoiding the full archive scan that failed during live reader QA. Existing path validation, ranges, ETags and legacy aliases are preserved.

Published snapshot `da0ccff91f2edda0abbccfa3087f505e19a9b2246a2d1862b02e2922751e663e`:70,162 total PDFs,28,699 ICLR2027 PDFs. First-page payload23,505bytes uncompressed/8,210compressed. Live measured list response0.488seconds initially/0.176seconds repeat, versus14.039seconds for the old catalogue alone (plus9.18MB compressed discovery metadata). A cold filtered request completed2.049seconds. These are API measurements, not guaranteed full-page timings.51 app tests pass; production build/Python compilation pass. Full typecheck retains pre-existing private-results.tsx errors. This supersedes the earlier browser-side full-metadata merge architecture for discovery browsing.

## Discovery redesign — October 3, 2026

Deployed version `b5bb1c5c-ac6d-40d2-9fb4-d4489570c8fb` (previous: `fac8fb03-8f31-49d4-8541-6e9dc6572478`) restyles only `components/paper-browser.tsx` and `.css`; data loading, URLs, snapshot fast path and the classifications view are unchanged. Results are compact rows with a monospace submission-number gutter (echoing the review template's line numbers); highlighter yellow is the single accent for search matches, active filters, saved papers and the current PDF page. Sticky header, results bar and area sidebar. The reader is two columns: a scrollable details column and a full-height embedded PDF (its duplicate heading is hidden via `.pb .pr-embedded` overrides). Titles strip common LaTeX/Markdown markup for display. `/` focuses search. System font stacks only, because the CSP restricts fonts to `'self'`. 51 tests pass; lint output is identical to the previous file's pre-existing Next.js rule hits.

## Classifications view on the shared design — October 3, 2026

Deployed version `9e7413e2-98fd-444e-8583-926dd91e4368` (previous: `b5bb1c5c-ac6d-40d2-9fb4-d4489570c8fb`). `components/site-chrome.tsx` (header, footer, reading-list count) and `components/site.css` (tokens, controls, reader error state) are now shared by Discover and `?view=classifications`; every public root carries `.site`. The classifications page dropped its separate terracotta "pangram" branding and illustrated hero, and uses the same type, gutter rows and sticky results bar. Its score colors remain green/amber/red data colors matching the PDF highlights. `PaperAtlas` takes `browse` (set only in `atlas-cloud/main.tsx`) to show Discover/Reading list links; the private local build omits them. Discover headline is now the collection name ("ICLR 2027 submissions"), with the subline "Search, filter by research area, and read the PDF."

## Mobile layout pass — October 3, 2026

Deployed version `16c18bd0-13d7-4868-8067-3b04c73a5ccf` (previous: `9e7413e2-98fd-444e-8583-926dd91e4368`). Fixed a live phone bug where Discover scrolled sideways: `.site` is a column flexbox, so `.site>main` needs `width:100%;min-width:0`. Compacted the mobile reader so the PDF starts sooner (keywords on one scrollable line, tighter spacing, equal-width action buttons), made the classifications dashboard pies small and side by side, and showed reader headings as "ICLR 2026" instead of `iclr/2026`. The deploy also includes the "no tldr found" icons that a concurrent Codex session added to paper-browser; their tooltip is now `display:none` until hover or focus, so it no longer widens the page.
