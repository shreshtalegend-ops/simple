from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        # Stores all active WebSocket connections
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        # Send a JSON message to all connected clients
        for connection in self.active_connections:
            await connection.send_json(message)

# Instantiate the global manager
manager = ConnectionManager()