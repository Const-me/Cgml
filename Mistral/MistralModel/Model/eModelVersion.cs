namespace Mistral;

/// <summary>Version of the model</summary>
/// <remarks>There're non-trivial differences between these models.<br/>
/// Sadly, I couldn't contain these incompatibilities within this DLL because the chat template is slightly different.</remarks>
enum eModelVersion: int
{
	/// <summary>Ported from MistralAI</summary>
	/// <seealso href="https://github.com/mistralai/mistral-src" />
	Original = 0,

	/// <summary>Ported from Hugging Face transformers 4.36.2</summary>
	/// <seealso href="https://github.com/huggingface/transformers/" />
	Instruct02 = 1
}