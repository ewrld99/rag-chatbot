import sys
import traceback
try:
    from app.core.config import settings
    print(f"DATABASE_URL is: {settings.DATABASE_URL}")
except Exception as e:
    traceback.print_exc()
