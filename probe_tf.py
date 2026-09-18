import traceback
try:
    from transformers import CLIPModel, CLIPProcessor
    print('OK')
except Exception:
    traceback.print_exc()
