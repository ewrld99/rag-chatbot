import requests
import json

API_URL = "http://localhost:8000/api/admin"

def test_validation(key, value, expected_status):
    print(f"Testing {key} = {value} (expecting {expected_status})")
    res = requests.put(f"{API_URL}/settings/{key}", json={"value": str(value)})
    if res.status_code == expected_status:
        print("✅ Pass")
    else:
        print(f"❌ Fail: got {res.status_code} ({res.text})")

def main():
    print("Running Settings Validation Tests...")
    # chunk_size (500-1500)
    test_validation("chunk_size", 400, 400)
    test_validation("chunk_size", 1600, 400)
    test_validation("chunk_size", 1000, 200)

    # top_k_dense (1-50)
    test_validation("top_k_dense", 0, 400)
    test_validation("top_k_dense", 51, 400)
    test_validation("top_k_dense", 20, 200)
    
    # allowed_extensions (comma separated list)
    test_validation("allowed_extensions", "pdf, docx, txt", 200)
    test_validation("allowed_extensions", "pdf!", 400)
    
    # fixing the embedding model for jina
    test_validation("embedding_model", "jina-embeddings-v2-base-en", 200)
    
if __name__ == "__main__":
    main()
