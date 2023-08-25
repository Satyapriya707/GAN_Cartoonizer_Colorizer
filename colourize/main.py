from fastapi import FastAPI
import uvicorn
import VideoColorizerRouter

app = FastAPI()

app.include_router(VideoColorizerRouter.router)



if __name__ == '__main__':
    uvicorn.run("main:app", port=8080, host='127.0.0.1', reload=True)