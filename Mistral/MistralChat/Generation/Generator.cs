namespace MistralChat;
using Mistral;
using MistralChat.ViewModels;
using System.Diagnostics;

/// <summary>Base class for both generator implementations</summary>
abstract class Generator: ChatClient
{
	protected readonly PromptBuilder builder = new PromptBuilder();

	public const string prompt = @"A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.";

	protected volatile bool shouldCancel = false;

	public override string? tryMakeResponse( string text )
	{
		if( shouldCancel )
			return text;

		int idx = text.IndexOf( "USER:", StringComparison.OrdinalIgnoreCase );
		if( idx > 0 )
			return text.Substring( 0, idx ).Trim();

		// It seems sometimes the model uses "</s>" to mark end of string.
		// It doesn’t always encode it as the EOS token.
		idx = text.IndexOf( "</s>", StringComparison.OrdinalIgnoreCase );
		if( idx > 0 )
			return text.Substring( 0, idx ).Trim();

		m_pending.setText( text );
		return null;
	}

	public void cancel() => shouldCancel = true;

	protected readonly Stopwatch stopwatch = new Stopwatch();

	public GeneratorStats? stats { get; private set; }

	public override void complete( int tokens )
	{
		stopwatch.Stop();
		stats = new GeneratorStats
		{
			tokens = tokens,
			tokensPerSecond = tokens / stopwatch.Elapsed.TotalSeconds,
		};
	}

	public bool disableRandomness;

	public override Random? random()
	{
		if( disableRandomness )
			return new Random( 25 );
		return null;
	}

	readonly PendingChatMessageVM m_pending = new PendingChatMessageVM();

	public iChatMessageVM pendingMessage()
	{
		m_pending.setText();
		return m_pending;
	}

	/// <summary>Reshape the <c>[ 0 .. count-1 ]</c> slice of the conversation history into a sequence of [ user, bot ] string tuples</summary>
	protected static IEnumerable<(string, string)> reshapeHistory( IReadOnlyList<iChatMessageVM> messages, int count )
	{
		for( int i = 0; i < count; i += 2 )
		{
			iChatMessageVM user = messages[ i ];
			iChatMessageVM bot = messages[ i + 1 ];
			Debug.Assert( user.user );
			Debug.Assert( !bot.user );
			yield return (user.text, bot.text);
		}
	}

	/// <summary>Generate response for chat mode</summary>
	public abstract Task<string> generateChat( iModel model, string initialPrompt, IReadOnlyList<iChatMessageVM> messages );

	protected string generateCompleteChat( iModel model, string initialPrompt, IReadOnlyList<iChatMessageVM> messages )
	{
		// Destroy old KV caches, and reset the positions
		model.stateRestore( null );

		// Build input tokens
		int count = messages.Count;
		if( 0 != count % 2 || count < 2 )
			throw new ApplicationException();

		IReadOnlyList<int> tokens;
		if( count > 2 )
		{
			// We actually have a history
			var history = reshapeHistory( messages, count - 2 );
			string last = messages[ count - 2 ].text;
			tokens = builder.complete( model.tokenizer, history, last, initialPrompt );
		}
		else
		{
			// No history is accumulated so far, generating a first message here
			Debug.Assert( count == 2 );
			Debug.Assert( messages[ 0 ].user );
			Debug.Assert( !messages[ 1 ].user );
			tokens = builder.single( model.tokenizer, messages[ 0 ].text, initialPrompt );
		}

		// Run the inference
		shouldCancel = false;
		stopwatch.Restart();
		return model.generate( tokens, this );
	}

	/// <summary>Generate a single response for text generation mode</summary>
	public Task<string> generateText( iModel model, string text )
	{
		string impl()
		{
			// Destroy old KV caches, and reset the positions
			model.stateRestore( null );

			// Build input tokens
			IReadOnlyList<int> tokens = builder.single( model.tokenizer, text );

			// Run the inference
			shouldCancel = false;
			stopwatch.Restart();
			return model.generate( tokens, this );
		}

		return Task.Run( impl );
	}
}