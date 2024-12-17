from fastapi import FastAPI
from app.api import router
import uvicorn
from starlette.responses import RedirectResponse
app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
