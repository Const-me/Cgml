namespace Torch;
using Cgml;

/// <summary>Load tensors from Python format to VRAM</summary>
public interface iWeightsLoader: IDisposable
{
	/// <summary>Loaded tensors in VRAM</summary>
	Dictionary<string, iTensor> tensors { get; }

	/// <summary>Load tensors from a single ZIP archive</summary>
	void loadSingle( string zip );

	/// <summary>Merge tensors from multiple ZIP archives</summary>
	void loadMultipart( string[] sources );
}