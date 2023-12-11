namespace MistralChat;
using Mistral;
using System.Text;

/// <summary>Utility class to generate chat prompts for Mistral</summary>
/// <seealso href="https://www.promptingguide.ai/models/mistral-7b#chat-template-for-mistral-7b-instruct" />
sealed class PromptBuilder
{
	readonly StringBuilder sb = new StringBuilder();
	readonly List<int> tokens = new List<int>();

	/// <summary>Encode a prompt with a single message</summary>
	public IReadOnlyList<int> single( iTokenizer tokenizer, string text, string? initialPrompt = null )
	{
		sb.Clear();
		sb.Append( "[INST]" );
		if( !string.IsNullOrWhiteSpace( initialPrompt ) )
		{
			sb.Append( initialPrompt );
			sb.Append( ' ' );
		}
		sb.Append( text );
		sb.Append( "[/INST]" );

		tokens.Clear();
		tokenizer.encode( tokens, sb.ToString() );
		return tokens;
	}

	/// <summary>Encode prompt with conversation history, and one last message</summary>
	/// <remarks>The history parameter is a sequence of [ user, bot ] string tuples.</remarks>
	public IReadOnlyList<int> complete( iTokenizer tokenizer, IEnumerable<(string, string)> history, string last, string? initialPrompt = null )
	{
		if( !history.Any() )
			return single( tokenizer, last, initialPrompt );

		tokens.Clear();
		tokens.Add( tokenizer.idBOS );

		sb.Clear();
		foreach( (string user, string bot) in history )
		{
			sb.Append( "[INST]" );

			if( !string.IsNullOrWhiteSpace( initialPrompt ) )
			{
				sb.Append( initialPrompt );
				sb.Append( ' ' );
				initialPrompt = null;
			}

			sb.Append( user );
			sb.Append( "[/INST]" );
			sb.Append( bot );
		}
		tokenizer.encode( tokens, sb.ToString() );

		tokens.Add( tokenizer.idEOS );

		sb.Clear();
		sb.Append( "[INST]" );
		sb.Append( last );
		sb.Append( "[/INST]" );
		tokenizer.encode( tokens, sb.ToString() );

		return tokens;
	}
}