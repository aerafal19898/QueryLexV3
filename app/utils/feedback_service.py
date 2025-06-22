"""
Supabase-based feedback service for collecting and processing user feedback.
"""

import uuid
import time
import re
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from app.utils.supabase_client import SupabaseVectorClient


class SupabaseFeedbackService:
    """Manage user feedback using Supabase database."""
    
    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: int = 587,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
        sender_email: Optional[str] = None,
        recipient_email: Optional[str] = None,
        use_tls: bool = True
    ):
        """Initialize the Supabase feedback service.
        
        Args:
            smtp_server: SMTP server for sending email
            smtp_port: SMTP port
            smtp_username: SMTP username
            smtp_password: SMTP password
            sender_email: Sender email address
            recipient_email: Recipient email address for feedback notifications
            use_tls: Whether to use TLS for SMTP connection
        """
        # Initialize Supabase client
        supabase_client = SupabaseVectorClient(use_service_key=True)
        self.client = supabase_client.client  # Access the raw Supabase client
        
        # Set up email configuration
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.sender_email = sender_email
        self.recipient_email = recipient_email
        self.use_tls = use_tls
        
        # Set up logger
        self.logger = logging.getLogger("feedback")
        
    def submit_feedback(
        self,
        user_id: Optional[str],
        feedback_type: str,
        content: str,
        rating: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        send_notification: bool = True
    ) -> str:
        """Submit feedback.
        
        Args:
            user_id: ID of the user submitting feedback
            feedback_type: Type of feedback (bug, feature, general, etc.)
            content: Feedback text content
            rating: Optional numerical rating (e.g., 1-5)
            metadata: Additional metadata about the feedback
            send_notification: Whether to send an email notification
            
        Returns:
            Feedback ID
        """
        try:
            # Generate feedback ID
            feedback_id = str(uuid.uuid4())
            
            # Create feedback data
            feedback_data = {
                "id": feedback_id,
                "user_id": user_id,
                "feedback_type": feedback_type,
                "content": content,
                "rating": rating,
                "metadata": metadata or {},
                "status": "new"
            }
            
            # Insert feedback into database
            result = self.client.table("feedback").insert(feedback_data).execute()
            
            if not result.data:
                raise Exception("Failed to insert feedback into database")
            
            # Log feedback submission
            self.logger.info(f"Feedback submitted: {feedback_id} - Type: {feedback_type} - User: {user_id}")
            
            # Send email notification if configured
            if send_notification and self.smtp_server and self.recipient_email:
                try:
                    feedback_with_timestamp = {**feedback_data, "created_at": datetime.now().isoformat()}
                    self._send_notification(feedback_with_timestamp)
                except Exception as e:
                    self.logger.error(f"Failed to send feedback notification: {e}")
            
            return feedback_id
            
        except Exception as e:
            self.logger.error(f"Error submitting feedback: {e}")
            raise
    
    def get_feedback(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        """Get feedback by ID.
        
        Args:
            feedback_id: Feedback ID
            
        Returns:
            Feedback data or None if not found
        """
        try:
            result = self.client.table("feedback").select("*").eq("id", feedback_id).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting feedback {feedback_id}: {e}")
            return None
    
    def update_feedback_status(
        self,
        feedback_id: str,
        status: str,
        notes: Optional[str] = None
    ) -> bool:
        """Update the status of feedback.
        
        Args:
            feedback_id: Feedback ID
            status: New status (new, in_progress, resolved, closed)
            notes: Optional notes about the status change
            
        Returns:
            Whether the update was successful
        """
        try:
            # Get current feedback to preserve status history
            current_feedback = self.get_feedback(feedback_id)
            if not current_feedback:
                return False
            
            # Get current status history or initialize empty list
            status_history = current_feedback.get("status_history", [])
            
            # Add new status change to history
            status_change = {
                "status": status,
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "notes": notes
            }
            status_history.append(status_change)
            
            # Update feedback with new status and history
            update_data = {
                "status": status,
                "status_history": status_history
            }
            
            result = self.client.table("feedback").update(update_data).eq("id", feedback_id).execute()
            
            if result.data:
                self.logger.info(f"Feedback status updated: {feedback_id} - Status: {status}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating feedback status {feedback_id}: {e}")
            return False
    
    def list_feedback(
        self,
        status: Optional[str] = None,
        feedback_type: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List feedback entries with optional filters.
        
        Args:
            status: Filter by status
            feedback_type: Filter by feedback type
            user_id: Filter by user ID
            start_time: Filter by start time (ISO format)
            end_time: Filter by end time (ISO format)
            limit: Maximum number of entries to return
            
        Returns:
            List of feedback entries
        """
        try:
            query = self.client.table("feedback").select("*")
            
            # Apply filters
            if status:
                query = query.eq("status", status)
            
            if feedback_type:
                query = query.eq("feedback_type", feedback_type)
            
            if user_id:
                query = query.eq("user_id", user_id)
            
            if start_time:
                query = query.gte("created_at", start_time)
            
            if end_time:
                query = query.lte("created_at", end_time)
            
            # Order by created_at descending and limit results
            query = query.order("created_at", desc=True).limit(limit)
            
            result = query.execute()
            return result.data or []
            
        except Exception as e:
            self.logger.error(f"Error listing feedback: {e}")
            return []
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get a summary of feedback statistics.
        
        Returns:
            Summary statistics
        """
        try:
            # Get all feedback
            all_feedback = self.client.table("feedback").select("*").execute()
            feedback_list = all_feedback.data or []
            
            summary = {
                "total": len(feedback_list),
                "by_status": {},
                "by_type": {},
                "by_rating": {},
                "recent": []
            }
            
            # Calculate 24 hours ago
            day_ago = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Process feedback entries
            for feedback in feedback_list:
                # Count by status
                status = feedback.get("status", "unknown")
                summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
                
                # Count by type
                feedback_type = feedback.get("feedback_type", "unknown")
                summary["by_type"][feedback_type] = summary["by_type"].get(feedback_type, 0) + 1
                
                # Count by rating
                rating = feedback.get("rating")
                if rating is not None:
                    rating_str = str(rating)
                    summary["by_rating"][rating_str] = summary["by_rating"].get(rating_str, 0) + 1
                
                # Add to recent list if created within last 24 hours
                created_at = feedback.get("created_at")
                if created_at:
                    try:
                        created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        if created_date >= day_ago:
                            summary["recent"].append({
                                "id": feedback.get("id"),
                                "feedback_type": feedback_type,
                                "rating": rating,
                                "status": status,
                                "created_at": created_at,
                                "user_id": feedback.get("user_id")
                            })
                    except (ValueError, TypeError):
                        continue
            
            # Sort recent feedback by created_at
            summary["recent"].sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting feedback summary: {e}")
            return {
                "total": 0,
                "by_status": {},
                "by_type": {},
                "by_rating": {},
                "recent": []
            }
    
    def delete_feedback(self, feedback_id: str) -> bool:
        """Delete feedback by ID.
        
        Args:
            feedback_id: Feedback ID
            
        Returns:
            Whether the deletion was successful
        """
        try:
            result = self.client.table("feedback").delete().eq("id", feedback_id).execute()
            
            if result.data:
                self.logger.info(f"Feedback deleted: {feedback_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error deleting feedback {feedback_id}: {e}")
            return False
    
    def _send_notification(self, feedback: Dict[str, Any]) -> bool:
        """Send an email notification about new feedback.
        
        Args:
            feedback: Feedback data
            
        Returns:
            Whether the email was sent successfully
        """
        if not self.smtp_server or not self.recipient_email:
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.sender_email
            msg["To"] = self.recipient_email
            msg["Subject"] = f"New QueryLex Feedback: {feedback.get('feedback_type')} - ID: {feedback.get('id')}"
            
            # Create email body
            body = f"""
            <h2>New Feedback Submission</h2>
            
            <p><strong>ID:</strong> {feedback.get('id')}</p>
            <p><strong>Type:</strong> {feedback.get('feedback_type')}</p>
            <p><strong>User:</strong> {feedback.get('user_id', 'Anonymous')}</p>
            <p><strong>Rating:</strong> {feedback.get('rating', 'N/A')}</p>
            <p><strong>Date:</strong> {feedback.get('created_at')}</p>
            
            <h3>Content:</h3>
            <p>{feedback.get('content')}</p>
            
            <h3>Additional Information:</h3>
            <pre>{feedback.get('metadata', {})}</pre>
            """
            
            # Attach body
            msg.attach(MIMEText(body, "html"))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email notification: {e}")
            return False
    
    def sanitize_feedback(self, content: str) -> str:
        """Sanitize feedback content to prevent spam and malicious content.
        
        Args:
            content: Raw feedback content
            
        Returns:
            Sanitized content
        """
        # Basic sanitization
        sanitized = content.strip()
        
        # Remove excessive whitespace and line breaks
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Limit length
        if len(sanitized) > 5000:
            sanitized = sanitized[:5000]
        
        # Check for suspicious patterns
        spam_patterns = [
            r'https?://',  # URLs
            r'www\.',
            r'\b(?:viagra|cialis|casino|lottery|winner)\b',  # Common spam words
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email addresses
        ]
        
        # Flag content as potentially spam
        is_potential_spam = False
        for pattern in spam_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                is_potential_spam = True
                break
        
        # Add spam warning if detected
        if is_potential_spam:
            sanitized = "[POTENTIAL SPAM] " + sanitized
        
        return sanitized
    
    def validate_feedback(
        self,
        content: str,
        feedback_type: str,
        rating: Optional[int] = None
    ) -> Tuple[bool, str]:
        """Validate feedback before submission.
        
        Args:
            content: Feedback content
            feedback_type: Feedback type
            rating: Optional numerical rating
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for empty content
        if not content or len(content.strip()) < 5:
            return False, "Feedback content must be at least 5 characters."
        
        # Check content length
        if len(content) > 5000:
            return False, "Feedback content must be less than 5000 characters."
        
        # Validate feedback type
        valid_types = ["bug", "feature", "general", "suggestion", "other"]
        if feedback_type not in valid_types:
            return False, f"Invalid feedback type. Must be one of: {', '.join(valid_types)}."
        
        # Validate rating if provided
        if rating is not None:
            try:
                rating_val = int(rating)
                if rating_val < 1 or rating_val > 5:
                    return False, "Rating must be between 1 and 5."
            except (ValueError, TypeError):
                return False, "Rating must be a number between 1 and 5."
        
        # Check for potential spam
        sanitized = self.sanitize_feedback(content)
        if sanitized.startswith("[POTENTIAL SPAM]"):
            return False, "Feedback appears to contain spam or prohibited content."
        
        return True, ""