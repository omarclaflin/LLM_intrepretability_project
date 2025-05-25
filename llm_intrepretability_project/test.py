# save this as, e.g., test_ssl.py

import os
# FIX: Import load_dataset directly
from datasets import load_dataset # <--- ADDED THIS LINE
import datasets # Keep this line as well, as it's used for datasets.config
import traceback

# !!! TEMPORARILY DISABLE SSL VERIFICATION - DO NOT USE IN PRODUCTION !!!
os.environ['HF_DATASETS_INSECURE_SSL_CERT'] = '1'
print("SSL verification TEMPORARILY DISABLED.")
# os.environ['HF_HUB_DISABLE_SSL_CERT_VERIFICATION'] = '1' # Alternative for hub library

try:
    print("Attempting to load tiny dataset with SSL verification disabled...")
    # Use the problematic dataset name here
    dataset = load_dataset("hf-internal-testing/tiny-text", split="train", streaming=True) # <-- This call should now work

    print("Dataset loaded successfully with SSL verification disabled!")
    # Optional: Try to iterate over one item to be sure data streaming also works
    # print("Reading one item...")
    # for item in dataset.take(1):
    #     print(item)
    #     break
    # print("Successfully read one item.")


except Exception as e:
    print("\n--- Error caught ---")
    print(f"Error type: {type(e)}")
    print(f"Error message: {e}")
    print("\n--- Traceback ---")
    traceback.print_exc()
    print("-----------------")

finally:
    # !!! IMPORTANT: UNSET THE ENVIRONMENT VARIABLE IMMEDIATELY !!!
    if 'HF_DATASETS_INSECURE_SSL_CERT' in os.environ:
        del os.environ['HF_DATASETS_INSECURE_SSL_CERT']
    # if 'HF_HUB_DISABLE_SSL_CERT_VERIFICATION' in os.environ:
    #    del os.environ['HF_HUB_DISABLE_SSL_CERT_VERIFICATION']
    print("SSL verification TEMPORARILY RE-ENABLED.")

print("Test finished.")