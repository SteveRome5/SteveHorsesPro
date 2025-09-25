#!/usr/bin/env python3
"""
Racing API Endpoint Finder
Find the correct endpoints for your racing API based on your working code
"""

import requests
import json
import base64
import os

# Your credentials
RUSER = 'WQaKSMwgmG8GnbkHgvRRCT0V'
RPASS = 'McYBoQViXSPvlNcvxQi1Z1py'
API_BASE = "https://api.theracingapi.com"

def test_endpoint(path, params=None):
    """Test a single endpoint"""
    url = API_BASE + path + ("?" + "&".join(f"{k}={v}" for k, v in (params or {}).items()))
    
    # Basic auth like your working code
    tok = base64.b64encode(f"{RUSER}:{RPASS}".encode()).decode()
    headers = {
        "Authorization": "Basic " + tok,
        "User-Agent": "Mozilla/5.0"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        return {
            'status': response.status_code,
            'url': url,
            'content': response.text[:500] if response.text else '',
            'success': response.status_code == 200
        }
    except Exception as e:
        return {
            'status': 'ERROR',
            'url': url,
            'content': str(e)[:200],
            'success': False
        }

def main():
    print("Testing Racing API Endpoints...")
    print("=" * 50)
    
    # Based on your working code, test these patterns
    endpoints_to_test = [
        # From your working versions
        ("/v1/north-america/meets", {"start_date": "2025-09-24", "end_date": "2025-09-24"}),
        
        # Alternative patterns found in racing APIs
        ("/v1/meetings", {"date": "2025-09-24"}),
        ("/v1/meetings/today", {}),
        ("/v1/races/today", {}),
        ("/v1/racecards/today", {}),
        ("/v1/racecards", {"date": "2025-09-24"}),
        
        # Root endpoints
        ("/", {}),
        ("/v1", {}),
        ("/api/v1", {}),
        
        # North America specific (your working pattern)
        ("/v1/north-america", {}),
        ("/v1/north-america/meetings", {"date": "2025-09-24"}),
        ("/v1/north-america/racecards", {"date": "2025-09-24"}),
        
        # Try without version
        ("/meetings", {"date": "2025-09-24"}),
        ("/racecards", {"date": "2025-09-24"}),
        
        # Alternative date formats
        ("/v1/north-america/meets", {"date": "2025-09-24"}),
        ("/v1/north-america/meets", {"start_date": "20250924"}),
    ]
    
    successful_endpoints = []
    
    for endpoint, params in endpoints_to_test:
        print(f"\nTesting: {endpoint}")
        if params:
            print(f"Params: {params}")
        
        result = test_endpoint(endpoint, params)
        
        print(f"Status: {result['status']}")
        if result['success']:
            print("✅ SUCCESS!")
            successful_endpoints.append((endpoint, params, result))
            print(f"Response preview: {result['content'][:200]}...")
        else:
            print(f"❌ Failed: {result['content'][:100]}")
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    
    if successful_endpoints:
        print(f"\n✅ Found {len(successful_endpoints)} working endpoints:")
        for endpoint, params, result in successful_endpoints:
            print(f"  - {endpoint} {params}")
            
        print("\nTo fix your scripts, update the API calls to use these working endpoints.")
        
        # Show sample working call
        endpoint, params, result = successful_endpoints[0]
        print(f"\nSample working call:")
        print(f"  GET {API_BASE}{endpoint}")
        if params:
            print(f"  Params: {params}")
    else:
        print("\n❌ No working endpoints found.")
        print("This suggests either:")
        print("  1. The API service is completely down")
        print("  2. Your credentials have expired")
        print("  3. The API has moved to a different domain")
        
        print("\nNext steps:")
        print("  1. Check if theracingapi.com is still the right domain")
        print("  2. Contact the API provider to verify endpoints")
        print("  3. Consider using the mock data version while investigating")

if __name__ == "__main__":
    main()