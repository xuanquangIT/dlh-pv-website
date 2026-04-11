import sys
import os

from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'main', 'backend')))
from app.main import app
from app.core.security import create_access_token

client = TestClient(app)

# Create a mock token for admin
token = create_access_token(data={"sub": "admin_user", "role": "admin"})

cookies = {"access_token": token}

print("Testing /forecast/daily")
res1 = client.get("/forecast/daily", cookies=cookies)
print(res1.status_code)
print(res1.text[:200])

print("Testing /ml-training/monitoring")
res2 = client.get("/ml-training/monitoring", cookies=cookies)
print(res2.status_code)
print(res2.text[:200])

print("Testing /model-registry/models-list")
res3 = client.get("/model-registry/models-list", cookies=cookies)
print(res3.status_code)
print(res3.text[:200])
