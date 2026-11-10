import requests
import json

img_path = r'C:/Users/singh/.gemini/antigravity/brain/665a9fec-00da-48fe-b182-dbe26c3b7721/.user_uploaded/media_1794284686678_71030ded.jpg'
with open(img_path, 'rb') as f:
    files = {'file': ('toi_article_screenshot.jpg', f, 'image/jpeg')}
    res = requests.post('http://127.0.0.1:8000/api/v1/verify', files=files, data={'deep_scan': 'false'})

data = res.json()
print('STATUS CODE:', res.status_code)
print('VERDICT:', data.get('verdict'))
print('CONFIDENCE:', data.get('confidence_score'))
print('EXPLANATION:', data.get('explanation'))
print('MANIPULATION SCORE:', data.get('forensics', {}).get('manipulation_score'))
print('SYNTHETIC SCORE:', data.get('forensics', {}).get('synthetic_score'))
print('IS DOCUMENT SCREENSHOT:', data.get('forensics', {}).get('is_document_screenshot'))
print('TOP SIGNALS:')
for s in data.get('top_signals', []):
    print("  *", s['name'] + ":", s['description'])
