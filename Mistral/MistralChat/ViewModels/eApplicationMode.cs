namespace MistralChat.ViewModels;

/// <summary>Mode of operation, switchable under View menu in the main window</summary>
enum eApplicationMode: byte
{
	/// <summary>The application behaves like an AI chat bot</summary>
	Chat,

	/// <summary>The application consumes a prompt, and generates a single response message.<br />
	/// No initial prompt or previous conversation history are used.</summary>
	TextGenerator,
}