from models_API.app.server import app as application

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(application, host="0.0.0.0", port=8080)
