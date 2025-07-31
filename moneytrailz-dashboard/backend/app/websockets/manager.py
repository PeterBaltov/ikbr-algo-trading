from fastapi import WebSocket
from typing import List, Dict, Any
import json
import asyncio
from datetime import datetime, timezone

class ConnectionManager:
    """WebSocket connection manager for real-time dashboard updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_subscriptions: Dict[WebSocket, List[str]] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept and add a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.client_subscriptions[websocket] = []
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection.established",
            "message": "Connected to moneytrailz Dashboard",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "client_id": id(websocket)
        }, websocket)
        
        print(f"🔗 New WebSocket connection: {id(websocket)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.client_subscriptions:
            del self.client_subscriptions[websocket]
        
        print(f"🔌 WebSocket disconnected: {id(websocket)}")
    
    async def disconnect_all(self):
        """Disconnect all active connections"""
        for connection in self.active_connections.copy():
            try:
                await connection.close()
            except:
                pass
        self.active_connections.clear()
        self.client_subscriptions.clear()
        print("🔌 All WebSocket connections closed")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific WebSocket connection"""
        try:
            if websocket in self.active_connections:
                await websocket.send_text(json.dumps(message))
        except Exception as e:
            print(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        if not self.active_connections:
            return
        
        message_text = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_text)
            except Exception as e:
                print(f"Error broadcasting to {id(connection)}: {e}")
                disconnected.append(connection)
        
        # Remove failed connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_to_subscribers(self, channel: str, message: Dict[str, Any]):
        """Broadcast to clients subscribed to a specific channel"""
        if not self.active_connections:
            return
        
        message_text = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            if channel in self.client_subscriptions.get(connection, []):
                try:
                    await connection.send_text(message_text)
                except Exception as e:
                    print(f"Error broadcasting to subscriber {id(connection)}: {e}")
                    disconnected.append(connection)
        
        # Remove failed connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def subscribe_client(self, websocket: WebSocket, channels: List[str]):
        """Subscribe a client to specific channels"""
        if websocket in self.client_subscriptions:
            for channel in channels:
                if channel not in self.client_subscriptions[websocket]:
                    self.client_subscriptions[websocket].append(channel)
            
            await self.send_personal_message({
                "type": "subscription.updated",
                "subscriptions": self.client_subscriptions[websocket],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, websocket)
    
    async def unsubscribe_client(self, websocket: WebSocket, channels: List[str]):
        """Unsubscribe a client from specific channels"""
        if websocket in self.client_subscriptions:
            for channel in channels:
                if channel in self.client_subscriptions[websocket]:
                    self.client_subscriptions[websocket].remove(channel)
            
            await self.send_personal_message({
                "type": "subscription.updated",
                "subscriptions": self.client_subscriptions[websocket],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, websocket)
    
    def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)
    
    def get_client_info(self) -> List[Dict[str, Any]]:
        """Get information about all connected clients"""
        return [
            {
                "client_id": id(connection),
                "subscriptions": self.client_subscriptions.get(connection, []),
                "connected_at": datetime.now(timezone.utc).isoformat()  # This would need to be tracked properly
            }
            for connection in self.active_connections
        ]

# Global manager instance
manager = ConnectionManager() 
