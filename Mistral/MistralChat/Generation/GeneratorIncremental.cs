namespace MistralChat;
using Mistral;
using MistralChat.ViewModels;

/// <summary>Implements incremental inference</summary>
sealed class GeneratorIncremental: Generator
{
	int generatedCount = 0;

	public override Task<string> generateChat( iModel model, string initialPrompt, IReadOnlyList<iChatMessageVM> messages )
	{
		string impl()
		{
			int countMessages = messages.Count;
			if( countMessages != ( generatedCount * 2 ) + 2 )
			{
				// User has clicked "regenerate last message" button
				// Can't reuse model's internal state because the state includes the old version of that generated message
				// Run non-incremental inference on the complete history..
				string complete = generateCompleteChat( model, initialPrompt, messages );
				// ..and reset counter of the generated messages
				generatedCount = countMessages / 2;
				return complete;
			}

			string? ip = null;
			if( 0 == generatedCount )
				ip = initialPrompt;

			string newText = messages[ messages.Count - 2 ].text;
			IReadOnlyList<int> tokens = builder.single( model.tokenizer, newText, ip );

			// Run the inference
			shouldCancel = false;
			stopwatch.Restart();
			string response = model.generate( tokens, this );
			generatedCount++;
			return response;
		};

		return Task.Run( impl );
	}
}