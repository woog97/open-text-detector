#!/bin/bash
# Deploy the redirect that runs under the site's old Worker name, pangram-paper-atlas (see app/atlas-cloud/redirect-worker.ts).
# Run it as `npm run deploy:atlas-redirect` from app/, only after `npm run deploy:atlas` has put the site on the new
# Worker and it has been checked: this replaces the old Worker's site with the redirect.
# Same safety checks as deploy_atlas.sh: main only, app/ committed. Records "commit <sha>" as the deploy message.
set -euo pipefail
BRANCH=main
cd "$(git rev-parse --show-toplevel)"
branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" = "$BRANCH" ] || { echo "STOP: deploy from $BRANCH; this checkout is on $branch."; exit 1; }
dirty=$(git status --porcelain -- app)
[ -z "$dirty" ] || { echo "STOP: app/ has changes that are not committed:"; echo "$dirty"; exit 1; }
head=$(git rev-parse HEAD)
cd app
code=$(curl -s -o /dev/null -w '%{http_code}' https://open-text-detector.woog09.workers.dev/)
[ "$code" = 200 ] || { echo "STOP: the new site answered HTTP $code; deploy it first with npm run deploy:atlas."; exit 1; }
npx wrangler deploy --config wrangler.atlas-redirect.json --message "commit $head (redirect to open-text-detector)"
echo "Deployed the redirect from commit $head."
