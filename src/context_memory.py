import requests

# Replace with your actual token
token = "lip_qsLf2qNCn127E0uqFQNl"  # Replace this with your actual token

headers = {
    "Authorization": f"Bearer {token}"
}

try:
    response = requests.post(
        "https://lichess.org/api/bot/account/upgrade", 
        headers=headers
    )
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("SUCCESS! Your account has been upgraded to a bot account.")
    else:
        print("ERROR: Failed to upgrade account.")
        
except Exception as e:
    print(f"An error occurred: {e}")