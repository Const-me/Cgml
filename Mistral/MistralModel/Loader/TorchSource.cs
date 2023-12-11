namespace Mistral;
using Cgml;
using System.Runtime.InteropServices;

/// <summary>Source locations of the model being imported</summary>
[StructLayout( LayoutKind.Auto )]
public struct TorchSource
{
	/// <summary>Path to the <c>tokenizer.model</c> file</summary>
	/// <remarks>That binary file is in google's protocol buffer format</remarks>
	public string tokenizer;

	/// <summary>Path to the folder with the weights, and <c>params.json</c> configuration file.</summary>
	public string weights;

	/// <summary>Tensors compression</summary>
	public eTensorLayout compression;
}