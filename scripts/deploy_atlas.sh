#!/bin/bash
# Deploy the Paper Atlas safely. Run it as `npm run deploy:atlas` from app/ (or run this script directly).
#
# It refuses to deploy when:
#   1. the checkout is not on main (every deploy comes from one branch);
#   2. app/ has uncommitted or untracked changes (what is live must exist in git);
#   3. the commit currently live is not contained in this one (deploying would erase someone else's work: merge first).
# Each deploy records "commit <sha>" as its message, which is how check 3 finds the live commit next time.
#
# --first   allow a deploy when the live version has no recorded commit (only after a rollback or a manual deploy,
#           once you have checked that this commit contains everything that version had).
set -euo pipefail
BRANCH=main
cd "$(git rev-parse --show-toplevel)"

branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" = "$BRANCH" ] || { echo "STOP: deploy from $BRANCH; this checkout is on $branch."; exit 1; }
dirty=$(git status --porcelain -- app)
[ -z "$dirty" ] || { echo "STOP: app/ has changes that are not committed:"; echo "$dirty"; exit 1; }
head=$(git rev-parse HEAD)

cd app
live=$( (npx wrangler deployments status --config wrangler.atlas.json --json 2>/dev/null || true) | python3 -c '
import json, re, sys
raw = sys.stdin.read().strip()
try:
    message = json.loads(raw).get("annotations", {}).get("workers/message", "") if raw else ""
except ValueError:  # no deployments yet (a new Worker) or an error message instead of JSON
    message = ""
found = re.search(r"commit ([0-9a-f]{7,40})", message)
print(found.group(1) if found else "")')
if [ -z "$live" ]; then
  if [ "${1:-}" != "--first" ]; then
    echo "STOP: the live version has no recorded commit (a rollback or a manual deploy)."
    echo "Check that $head contains everything it had, then rerun with --first."
    exit 1
  fi
elif ! git cat-file -e "$live^{commit}" 2>/dev/null; then
  echo "STOP: live commit $live is not in this repository. Fetch or merge it first."; exit 1
elif ! git merge-base --is-ancestor "$live" "$head"; then
  echo "STOP: the live site runs commit $live, which $head does not include."
  echo "Deploying would remove that work. Merge it into $BRANCH first: git merge $live"
  exit 1
fi

npm run build:atlas
npx wrangler deploy --config wrangler.atlas.json --message "commit $head"
echo "Deployed commit $head. Live before: ${live:-unrecorded}."
