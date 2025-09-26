import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore
from long_memory import call_model


class TestCallModel(unittest.TestCase):
    """
    Unit tests for the `call_model` function in long_memory.py.
    """

    def setUp(self):
        """
        Setup mock objects and test data.
        """
        self.mock_store = MagicMock(spec=BaseStore)
        self.mock_state = MagicMock(spec=MessagesState)
        self.mock_config = {"configurable": {"user_id": "1"}}
        self.mock_message = MagicMock()
        self.mock_message.content = "Test message"
        self.mock_state.messages = [self.mock_message]

    def test_call_model_without_remember(self):
        """
        Test `call_model` when the user does not ask to remember anything.
        """
        self.mock_store.search.return_value = []
        result = call_model(self.mock_state, self.mock_config, store=self.mock_store)
        self.assertIn("messages", result)
        self.mock_store.search.assert_called_once()
        self.mock_store.put.assert_not_called()

    def test_call_model_with_remember(self):
        """
        Test `call_model` when the user asks to remember something.
        """
        self.mock_message.content = "Remember: my name is Bob"
        self.mock_store.search.return_value = []
        result = call_model(self.mock_state, self.mock_config, store=self.mock_store)
        self.assertIn("messages", result)
        self.mock_store.search.assert_called_once()
        self.mock_store.put.assert_called_once()

    def test_call_model_with_existing_memories(self):
        """
        Test `call_model` when the user has existing memories.
        """
        mock_memory = MagicMock()
        mock_memory.value = {"data": "User name is Bob"}
        self.mock_store.search.return_value = [mock_memory]
        result = call_model(self.mock_state, self.mock_config, store=self.mock_store)
        self.assertIn("messages", result)
        self.mock_store.search.assert_called_once()
        self.mock_store.put.assert_not_called()

    def test_call_model_with_empty_messages(self):
        """
        Test `call_model` when the messages list is empty.
        """
        self.mock_state.messages = []
        with self.assertRaises(IndexError):
            call_model(self.mock_state, self.mock_config, store=self.mock_store)


if __name__ == "__main__":
    unittest.main()