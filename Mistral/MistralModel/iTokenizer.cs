namespace Mistral;

/// <summary>Interface to access the tokenizer</summary>
public interface iTokenizer
{
	/// <summary>Encode string, and append tokens to the list</summary>
	void encode( List<int> tokens, string text );

	/// <summary>ID of the BOS <c>&lt;s&gt;</c> token.</summary>
	int idBOS { get; }

	/// <summary>ID of the EOS <c>&lt;/s&gt;</c> token.</summary>
	int idEOS { get; }
}