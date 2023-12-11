namespace MistralChat;
using Mistral;
using MistralChat.ViewModels;
using System.Collections.Generic;
using System.Threading.Tasks;

/// <summary>This implementation destroys model's internal state before generating new messages, and feeds the complete history to the model.</summary>
/// <remarks>The functionality closely mimics the original chat template.<br/>
/// However, the performance degrades linearly with the length of the chat history.</remarks>
sealed class GeneratorComplete: Generator
{
	public override Task<string> generateChat( iModel model, string initialPrompt, IReadOnlyList<iChatMessageVM> messages )
	{
		Func<string> impl = () => generateCompleteChat( model, initialPrompt, messages );
		return Task.Run( impl );
	}
}