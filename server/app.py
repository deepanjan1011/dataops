"""Thin wrapper that re-exports the FastAPI app for OpenEnv spec compliance."""
from dataops_gym.server.app import app  # noqa: F401


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
