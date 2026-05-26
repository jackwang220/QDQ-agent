"""Entry point: python -m qdq_agent.server [--host HOST] [--port PORT]"""
import argparse
import uvicorn

parser = argparse.ArgumentParser(description="QDQ Agent web panel")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

uvicorn.run("qdq_agent.server.app:app", host=args.host, port=args.port, reload=False)
