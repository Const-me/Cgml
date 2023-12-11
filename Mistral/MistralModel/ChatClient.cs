namespace Mistral;

/// <summary>Client for an interactive chat session</summary>
public abstract class ChatClient
{
	/// <summary>Test whether the string has a stop marker.<br/>
	/// If it has, return chat response.<br/>
	/// Otherwise, return null and the model will generate and then decode moar tokens</summary>
	public virtual string? tryMakeResponse( string text ) => null;

	/// <summary>Maximum count of response tokens to generate</summary>
	public virtual int maxResponseTokens() => 512;

	/// <summary>Called when the generation is complete</summary>
	public virtual void complete( int tokens ) { }

	/// <summary>Custom random number generator</summary>
	public virtual Random? random() => null;
}