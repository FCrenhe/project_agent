import unittest
from unittest.mock import patch, MagicMock
from main import stream_graph_updates, BasicToolNode, State

class TestStreamGraphUpdates(unittest.TestCase):
    """
    Test suite for the stream_graph_updates function.
    This function streams updates from a graph based on user input.
    """

    @patch('main.graph')
    def test_stream_graph_updates_normal_input(self, mock_graph):
        """
        Test that stream_graph_updates processes normal user input correctly.
        """
        # Setup mock
        mock_chunk = MagicMock()
        mock_chunk[0].content = "Test response"
        mock_graph.stream.return_value = [mock_chunk]

        # Call function
        stream_graph_updates("Test input")

        # Assertions
        mock_graph.stream.assert_called_once_with(
            {"messages": [{"role": "user", "content": "Test input"}]}, stream_mode="messages"
        )

    @patch('main.graph')
    def test_stream_graph_updates_empty_input(self, mock_graph):
        """
        Test that stream_graph_updates handles empty input gracefully.
        """
        # Setup mock
        mock_chunk = MagicMock()
        mock_chunk[0].content = "Test response"
        mock_graph.stream.return_value = [mock_chunk]

        # Call function
        stream_graph_updates("")

        # Assertions
        mock_graph.stream.assert_called_once_with(
            {"messages": [{"role": "user", "content": ""}]}, stream_mode="messages"
        )

    @patch('main.graph')
    def test_stream_graph_updates_error_handling(self, mock_graph):
        """
        Test that stream_graph_updates handles errors during streaming.
        """
        # Setup mock to raise an exception
        mock_graph.stream.side_effect = Exception("Streaming error")

        # Call function and verify it doesn't crash
        stream_graph_updates("Test input")


class TestBasicToolNode(unittest.TestCase):
    """
    Test suite for the BasicToolNode class.
    This class processes tool calls in messages.
    """

    def setUp(self):
        """
        Setup test environment with a mock tool.
        """
        self.tool = MagicMock()
        self.tool.name = "test_tool"
        self.tool.invoke.return_value = "Test result"
        self.node = BasicToolNode(tools=[self.tool])

    def test_basic_tool_node_with_tool_calls(self):
        """
        Test BasicToolNode processes tool calls correctly.
        """
        # Setup input with tool calls
        inputs = {
            "messages": [
                MagicMock(
                    tool_calls=[{"name": "test_tool", "args": {}, "id": "1"}]
                )
            ]
        }

        # Call function
        result = self.node(inputs)

        # Assertions
        self.tool.invoke.assert_called_once_with({})
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0].content, '"Test result"')

    def test_basic_tool_node_no_tool_calls(self):
        """
        Test BasicToolNode raises ValueError if no tool calls are found.
        """
        # Setup input without tool calls
        inputs = {"messages": [MagicMock(tool_calls=[])]}

        # Call function and verify exception
        with self.assertRaises(ValueError):
            self.node(inputs)


if __name__ == "__main__":
    unittest.main()