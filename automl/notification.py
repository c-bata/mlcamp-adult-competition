import os
import json
import requests


def notify_slack(msg: str):
    url = os.getenv("WEBHOOK_URL", None)
    channel = os.getenv('WEBHOOK_SLACK_CHANNEL', None)

    if url is None or channel is None:
        print(msg)
        return

    requests.post(url, data=json.dumps({
        'channel': channel,
        'text': msg,
        'username': 'ML camp server',
        'link_names': 1,
    }))
