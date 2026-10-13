"""Run locally once to enable automatic OpenReview login."""
import getpass
import json
from pathlib import Path
import subprocess

email = input('OpenReview email: ').strip()
password = getpass.getpass('OpenReview password (saved in macOS Keychain): ')
if not email or not password:
    raise SystemExit('Email and password must not be empty.')
result = subprocess.run(
    ['security', 'add-generic-password', '-U', '-a', email,
     '-s', 'pangram-openreview', '-w', password],
    capture_output=True, text=True)
password = None
if result.returncode:
    raise SystemExit('Could not save the password in Keychain.')
account_file = Path.home() / '.config/open-text-detector/openreview-account.json'
account_file.parent.mkdir(parents=True, exist_ok=True)
account_file.write_text(json.dumps({'email': email}) + '\n')
account_file.chmod(0o600)
print('Saved. The existing download command will now use Keychain automatically.')
