namespace Mistral.Model;

/// <summary>Define square diagonal mask</summary>
/// <remarks>The mask has 0.0 in the lower-left corner, and -inf in the top-right corner</remarks>
readonly struct ModelMask
{
	/// <summary>Size of the mask in elements</summary>
	public readonly int size;

	/// <summary>If diagonal = 0, all elements on and above the main diagonal are -inf.<br />
	/// A positive value excludes just as many diagonals above the main diagonal</summary>
	public readonly int diagonal;

	public ModelMask( int colStart, int colEnd )
	{
		size = colEnd - colStart;
		diagonal = colStart + 1;
	}

	/// <summary>A string for debugger</summary>
	public override string ToString() => $"Size {size}, diagonal {diagonal}";
}