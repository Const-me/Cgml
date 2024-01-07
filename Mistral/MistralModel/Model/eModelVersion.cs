namespace Mistral;

/// <summary>Version of the model</summary>
public enum eModelVersion: int
{
	/// <summary>Ported from MistralAI</summary>
	/// <seealso href="https://github.com/mistralai/mistral-src" />
	Original = 0,

	/// <summary>Ported from Hugging Face transformers 4.36.2</summary>
	/// <seealso href="https://github.com/huggingface/transformers/" />
	Instruct02 = 1
}