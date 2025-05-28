#!/usr/bin/env python
"""
Test script for custom instructions functionality.
"""

import os
import sys
import json
import requests
from pathlib import Path

print("Testing Custom Instructions")
print("-------------------------------------")

def test_custom_instructions():
    # Base URL
    base_url = "http://localhost:5000"
    
    try:
        # 1. Create a test dataset with custom instructions
        print("\n1. Creating test dataset...")
        dataset_data = {
            "name": "test-custom-instructions",
            "description": "Test dataset for custom instructions",
            "custom_instructions": "You are a casual assistant. Respond in a friendly, informal tone. No need for formal sections or structure. Just be helpful and conversational."
        }
        
        response = requests.post(f"{base_url}/api/datasets", json=dataset_data)
        print("Create dataset response:", response.json())
        
        # 2. Create a new chat with this dataset
        print("\n2. Creating new chat...")
        chat_data = {
            "title": "Test Custom Instructions",
            "dataset": "test-custom-instructions"
        }
        
        response = requests.post(f"{base_url}/api/chats", json=chat_data)
        print("Create chat response:", response.json())
        chat_id = response.json()["id"]
        print("Created chat with ID:", chat_id)
        
        # 3. Send a test message
        print("\n3. Sending test message...")
        message_data = {
            "message": "What are the key points about EU sanctions?",
            "dataset": "test-custom-instructions"
        }
        
        response = requests.post(f"{base_url}/api/chats/{chat_id}/messages", json=message_data)
        print("\nResponse from chat:")
        print(response.text)
        
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: {str(e)}")
        print("Make sure the Flask server is running on http://localhost:5000")
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_custom_instructions() 