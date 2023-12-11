namespace MistralChat;
using System.Runtime.InteropServices;

/// <summary>High-level performance stats from the generation of 1 chat response</summary>
[StructLayout( LayoutKind.Auto )]
readonly struct GeneratorStats
{
	/// <summary>Count of tokens produced</summary>
	/// <remarks>The number includes stop terminator, user sees slightly fewer</remarks>
	public int tokens { get; init; }

	/// <summary>Average count of tokens generated per second</summary>
	public double tokensPerSecond { get; init; }

	/// <summary>Produce a string for the GUI</summary>
	public override string ToString() =>
		$"{tokens} tokens, {tokensPerSecond:0.###} tokens per second";
}