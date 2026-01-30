#!/usr/bin/env python3
import requests
import argparse
import sys
import json
import os

def scan_cvs(directory_path, api_url):
    """
    Trigger batch processing of CVs in a directory via API.
    """
    url = f"{api_url}/api/v1/cv/scan"
    
    # Resolve absolute path for clarity
    abs_path = os.path.abspath(directory_path)
    print(f"Scanning directory: {abs_path}")
    
    payload = {
        "directory_path": abs_path
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print("\n✅ Batch Processing Started Successfully!")
        print(f"----------------------------------------")
        print(f"Directory: {result.get('directory')}")
        print(f"PDFs Found: {result.get('found_files')}")
        print(f"Tasks Triggered: {result.get('triggered_tasks')}")
        
        if result.get('errors'):
            print(f"\n⚠️ Errors ({len(result['errors'])}):")
            for err in result['errors']:
                print(f"  - {err}")
                
        print("\nMonitor logs using: celery -A app.core.celery_app worker -l info")
                
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Error: Could not connect to API at {api_url}")
        print("Is the server running? (Try: ./start_project.sh)")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ API Error: {e}")
        try:
            print(f"Details: {response.text}")
        except:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch scan and process CVs from a directory")
    parser.add_argument("directory", help="Path to the directory containing PDF CVs")
    parser.add_argument("--url", default="http://localhost:8000", help="API Base URL (default: http://localhost:8000)")
    
    args = parser.parse_args()
    
    scan_cvs(args.directory, args.url)
