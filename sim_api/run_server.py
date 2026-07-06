import nest_asyncio
nest_asyncio.apply()

import uvicorn
import os
import sys

# Ensure sim_api is in path so we can import it correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
