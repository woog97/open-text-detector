"""Superseded: reader lookup shards are now published with the public catalogue. Runs publish_public_catalogue.py."""
from publish_public_catalogue import main

if __name__ == '__main__':
    import sys
    main(force='--force' in sys.argv)
