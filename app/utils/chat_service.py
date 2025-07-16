"""
Chat service for Supabase operations.
"""

import os
import json
import uuid
import time
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from app.config import SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_KEY
from app.utils.connection_retry import retry_on_connection_error, safe_supabase_call


class SupabaseChatService:
    """Service for managing chats and folders in Supabase."""
    
    def __init__(self, use_service_key: bool = False):
        """Initialize Supabase client.
        
        Args:
            use_service_key: Whether to use service key for admin operations
        """
        if not SUPABASE_URL:
            raise ValueError("SUPABASE_URL environment variable is required")
        
        key = SUPABASE_SERVICE_KEY if use_service_key and SUPABASE_SERVICE_KEY else SUPABASE_ANON_KEY
        if not key:
            raise ValueError("SUPABASE_ANON_KEY or SUPABASE_SERVICE_KEY environment variable is required")
        
        self.client: Client = create_client(SUPABASE_URL, key)
        self.use_service_key = use_service_key
    
    # Folder Operations
    def create_folder(self, name: str, folder_id: str = None) -> Dict[str, Any]:
        """Create a new chat folder.
        
        Args:
            name: Folder name
            folder_id: Optional custom folder ID
            
        Returns:
            Folder data
        """
        if not folder_id:
            folder_id = f"folder_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        folder_data = {
            "id": folder_id,
            "name": name
        }
        
        result = self.client.table("folders").insert(folder_data).execute()
        
        if result.data:
            return result.data[0]
        else:
            raise Exception(f"Failed to create folder: {result}")
    
    def get_folder(self, folder_id: str) -> Optional[Dict[str, Any]]:
        """Get folder by ID.
        
        Args:
            folder_id: Folder ID
            
        Returns:
            Folder data or None if not found
        """
        result = self.client.table("folders").select("*").eq("id", folder_id).execute()
        
        if result.data:
            return result.data[0]
        return None
    
    def list_folders(self) -> List[Dict[str, Any]]:
        """List all folders.
        
        Returns:
            List of folder data
        """
        result = self.client.table("folders").select("*").order("created_at").execute()
        return result.data or []
    
    def update_folder(self, folder_id: str, name: str) -> bool:
        """Update folder name.
        
        Args:
            folder_id: Folder ID
            name: New folder name
            
        Returns:
            True if successful
        """
        try:
            result = self.client.table("folders").update({"name": name}).eq("id", folder_id).execute()
            return bool(result.data)
        except Exception as e:
            print(f"Error updating folder {folder_id}: {e}")
            return False
    
    def delete_folder(self, folder_id: str) -> bool:
        """Delete a folder and move its chats to default folder.
        
        Args:
            folder_id: Folder ID
            
        Returns:
            True if successful
        """
        try:
            # Move chats to default folder
            self.client.table("chats").update({"folder_id": "default"}).eq("folder_id", folder_id).execute()
            
            # Delete the folder
            result = self.client.table("folders").delete().eq("id", folder_id).execute()
            
            return True
        except Exception as e:
            print(f"Error deleting folder {folder_id}: {e}")
            return False
    
    # Chat Operations
    def create_chat(self, title: str, folder_id: str = "default", dataset: str = None, 
                   model: str = None, chat_id: str = None) -> Dict[str, Any]:
        """Create a new chat.
        
        Args:
            title: Chat title
            folder_id: Folder ID (defaults to 'default')
            dataset: Dataset name
            model: Model name
            chat_id: Optional custom chat ID
            
        Returns:
            Chat data
        """
        if not chat_id:
            chat_id = f"chat_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        chat_data = {
            "id": chat_id,
            "title": title,
            "folder_id": folder_id,
            "dataset": dataset,
            "model": model
        }
        
        result = self.client.table("chats").insert(chat_data).execute()
        
        if result.data:
            return result.data[0]
        else:
            raise Exception(f"Failed to create chat: {result}")
    
    @retry_on_connection_error(max_retries=3, backoff_factor=0.5)
    def get_chat(self, chat_id: str, include_messages: bool = True) -> Optional[Dict[str, Any]]:
        """Get chat by ID.
        
        Args:
            chat_id: Chat ID
            include_messages: Whether to include messages
            
        Returns:
            Chat data with messages or None if not found
        """
        # Validate chat_id format
        if not chat_id or chat_id == '[object Object]' or not isinstance(chat_id, str):
            print(f"Invalid chat_id format: {chat_id}")
            return None
        
        # Get chat data
        result = self.client.table("chats").select("*").eq("id", chat_id).execute()
        
        if not result.data:
            return None
        
        chat = result.data[0]
        
        if include_messages:
            # Get messages
            messages_result = self.client.table("messages").select("*").eq("chat_id", chat_id).order("created_at").execute()
            chat["messages"] = messages_result.data or []
        
        return chat
    
    @retry_on_connection_error(max_retries=3, backoff_factor=0.5)
    def list_chats(self, folder_id: str = None) -> List[Dict[str, Any]]:
        """List chats, optionally filtered by folder.
        
        Args:
            folder_id: Optional folder ID to filter by
            
        Returns:
            List of chat data
        """
        query = self.client.table("chats").select("*")
        
        if folder_id:
            query = query.eq("folder_id", folder_id)
        
        result = query.order("created_at", desc=True).execute()
        return result.data or []
    
    def update_chat(self, chat_id: str, **kwargs) -> bool:
        """Update chat properties.
        
        Args:
            chat_id: Chat ID
            **kwargs: Properties to update
            
        Returns:
            True if successful
        """
        try:
            # Remove None values
            update_data = {k: v for k, v in kwargs.items() if v is not None}
            
            if not update_data:
                return True
            
            result = self.client.table("chats").update(update_data).eq("id", chat_id).execute()
            return bool(result.data)
        except Exception as e:
            print(f"Error updating chat {chat_id}: {e}")
            return False
    
    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat and all its messages.
        
        Args:
            chat_id: Chat ID
            
        Returns:
            True if successful
        """
        try:
            # Messages will be automatically deleted due to CASCADE constraint
            result = self.client.table("chats").delete().eq("id", chat_id).execute()
            
            return True
        except Exception as e:
            print(f"Error deleting chat {chat_id}: {e}")
            return False
    
    def move_chat_to_folder(self, chat_id: str, folder_id: str) -> bool:
        """Move chat to a different folder.
        
        Args:
            chat_id: Chat ID
            folder_id: Target folder ID
            
        Returns:
            True if successful
        """
        return self.update_chat(chat_id, folder_id=folder_id)
    
    # Message Operations
    @retry_on_connection_error(max_retries=3, backoff_factor=0.5)
    def add_message(self, chat_id: str, role: str, content: str, 
                   metadata: Dict[str, Any] = None, message_id: str = None) -> Dict[str, Any]:
        """Add a message to a chat.
        
        Args:
            chat_id: Chat ID
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional message metadata
            message_id: Optional custom message ID
            
        Returns:
            Message data
        """
        if not message_id:
            message_id = f"msg_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        message_data = {
            "id": message_id,
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        
        result = self.client.table("messages").insert(message_data).execute()
        
        if result.data:
            # Update chat updated_at timestamp
            self.client.table("chats").update({"updated_at": "NOW()"}).eq("id", chat_id).execute()
            return result.data[0]
        else:
            raise Exception(f"Failed to add message: {result}")
    
    @retry_on_connection_error(max_retries=3, backoff_factor=0.5)
    def get_messages(self, chat_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a chat.
        
        Args:
            chat_id: Chat ID
            
        Returns:
            List of message data
        """
        result = self.client.table("messages").select("*").eq("chat_id", chat_id).order("created_at").execute()
        return result.data or []
    
    def update_message(self, message_id: str, content: str = None, 
                      metadata: Dict[str, Any] = None) -> bool:
        """Update a message.
        
        Args:
            message_id: Message ID
            content: Optional new content
            metadata: Optional new metadata
            
        Returns:
            True if successful
        """
        try:
            update_data = {}
            if content is not None:
                update_data["content"] = content
            if metadata is not None:
                update_data["metadata"] = metadata
            
            if not update_data:
                return True
            
            result = self.client.table("messages").update(update_data).eq("id", message_id).execute()
            
            # Also update the chat's updated_at timestamp
            if result.data:
                # Get the chat_id for this message
                message_data = result.data[0] if result.data else None
                if message_data and message_data.get('chat_id'):
                    self.client.table("chats").update({"updated_at": "NOW()"}).eq("id", message_data['chat_id']).execute()
            
            return bool(result.data)
        except Exception as e:
            print(f"Error updating message {message_id}: {e}")
            return False
    
    def delete_message(self, message_id: str) -> bool:
        """Delete a message.
        
        Args:
            message_id: Message ID
            
        Returns:
            True if successful
        """
        try:
            result = self.client.table("messages").delete().eq("id", message_id).execute()
            return True
        except Exception as e:
            print(f"Error deleting message {message_id}: {e}")
            return False
    
    
    


# Create a global instance for backwards compatibility
chat_service = SupabaseChatService()