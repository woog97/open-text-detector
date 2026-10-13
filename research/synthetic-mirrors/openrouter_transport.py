"""Space-side API transport; key supplied in process memory, never persisted."""
import asyncio
import os
import requests

async def fetch(path, body=None, auth=True):
    def send():
        headers = {'Content-Type': 'application/json'}
        if auth:
            headers['Authorization'] = 'Bearer ' + os.environ['OPENROUTER_API_KEY']
        try:
            response = requests.request('POST' if body is not None else 'GET',
                'https://openrouter.ai/api/v1/' + path, headers=headers, json=body,
                timeout=(20, 900), allow_redirects=False)
            try:
                return response.status_code, response.json()
            except ValueError:
                return response.status_code, {'error': {'type': 'non_json'}}
        except requests.RequestException as exc:
            return 0, {'error': {'type': type(exc).__name__}}
    return await asyncio.to_thread(send)
