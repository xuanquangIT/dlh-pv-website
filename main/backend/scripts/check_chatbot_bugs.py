import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"
API_URL = f"{BASE_URL}/solar-ai-chat/query"

test_cases = [
    {
        "bug_id": "BUG-01",
        "description": "System overview pulling exact single latest row from mart_system_kpi_daily.",
        "query": "System overview",
        "role": "admin"
    },
    {
        "bug_id": "BUG-02",
        "description": "Energy performance should expose facility ID alongside name.",
        "query": "Cho tôi xem energy performance của các trạm",
        "role": "admin"
    },
    {
        "bug_id": "BUG-03 & BUG-04",
        "description": "Lookup FINLEYSF should work + WRSF1 should be 20MW.",
        "query": "Give me facility info for FINLEYSF and WRSF1",
        "role": "admin"
    },
    {
        "bug_id": "BUG-05",
        "description": "Time-Anchoring 'Yesterday' correctly offset against latest mart_energy_daily date.",
        "query": "Lượng energy cao nhất hôm qua",
        "role": "admin"
    }
]

def check_chatbot():
    print("=" * 60)
    print("Running Chatbot Integration Checks for Bug Fixes")
    print("=" * 60)
    
    session = requests.Session()
    
    try:
        # Authenticate first
        login_res = session.post(
            f"{BASE_URL}/auth/login",
            data={"username": "admin", "password": "admin123", "next": "/dashboard"},
            allow_redirects=False
        )
        if login_res.status_code not in (200, 302, 303):
            print(f"ERROR: Login failed with status {login_res.status_code}")
            return
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Connection Refused. Please start the FastAPI backend server (http://127.0.0.1:8000) first.")
        return
        
    for case in test_cases:
        print(f"\n[{case['bug_id']}] {case['description']}")
        print(f"User > {case['query']}")
        
        # Create a chat session ID 
        try:
            sess_res = session.post(
                f"{BASE_URL}/solar-ai-chat/sessions",
                json={"role": case['role'], "title": "Check script"}
            )
            chat_session_id = sess_res.json().get("session_id")
        except Exception:
            chat_session_id = ""
        
        payload = {
            "message": case['query'],
            "role": case['role'],
            "session_id": chat_session_id
        }
        
        try:
            start_time = time.time()
            response = session.post(API_URL, json=payload, timeout=30.0)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"Chatbot < (Topic: {data.get('topic')}, {latency:.2f}s)")
                print(f"Answer: {data.get('answer', '')[:300]}...")
                
                metrics = data.get('key_metrics', {})
                if metrics:
                    print(f"Metrics (Keys): {list(metrics.keys())}")
                    
            else:
                print(f"Error Code: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    check_chatbot()
