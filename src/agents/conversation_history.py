"""
Conversation history management for agents.
"""
from typing import Dict, List


class ConversationHistory:
    """
    Manages message history for an agent's conversation.

    Provides a clean interface for adding, retrieving, and resetting messages.
    """

    def __init__(self, system_prompt: str = ""):
        """
        Initialize conversation history.

        Args:
            system_prompt: Optional system prompt to start the conversation.
        """
        self.messages: List[Dict[str, str]] = []

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: Message role ("system", "user", "assistant").
            content: Message content.
        """
        self.messages.append({"role": role, "content": content})

    def delete_message(self) -> None:
        """Delete the last message from the conversation history."""
        self.messages = self.messages[:-1]

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message."""
        self.add_message("assistant", content)

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in the conversation.

        Returns:
            List of message dictionaries with "role" and "content" keys.
        """
        return self.messages

    def reset(self, keep_system_prompt: bool = True) -> None:
        """
        Reset the conversation history.

        Args:
            keep_system_prompt: If True, keep the initial system prompt.
        """
        if keep_system_prompt and self.messages and self.messages[0]["role"] == "system":
            system_msg = self.messages[0]
            self.messages = [system_msg]
        else:
            self.messages = []

    def reset_system_prompt(self, new_system_prompt: str) -> None:
        """
        Reset the system prompt in the conversation history.

        Args:
            new_system_prompt: The new system prompt to set.
        """
        self.reset(keep_system_prompt=False)
        self.add_message("system", new_system_prompt)

    def __len__(self) -> int:
        """Return the number of messages in history."""
        return len(self.messages)

    def __repr__(self) -> str:
        """String representation."""
        return f"ConversationHistory(messages={len(self.messages)})"