namespace MistralChat;
using System.IO;

sealed class MistralDir
{
	public readonly string statusMessage;
	public readonly string? dir;
	public readonly long? bytes;

	public const string tokenizer = "tokenizer.model";
	readonly string[] requiredFiles =
		new string[] { "consolidated.00.pth", "params.json", tokenizer };

	bool hasRequiredFiles( string dir, out long totalBytes )
	{
		totalBytes = 0;
		long cb = 0;
		foreach( string req in requiredFiles )
		{
			string path = Path.Combine( dir, req );
			if( !File.Exists( path ) )
				return false;
			cb += new FileInfo( path ).Length;
		}
		totalBytes = cb;
		return true;
	}

	public MistralDir( string dir )
	{
		if( string.IsNullOrEmpty( dir ) )
		{
			statusMessage = "No folder selected";
			return;
		}
		if( !Directory.Exists( dir ) )
		{
			statusMessage = "The folder doesn’t exist";
			return;
		}

		long bytes;
		if( !hasRequiredFiles( dir, out bytes ) )
		{
			statusMessage = "The folder doesn’t contain a Mistral model";
			return;
		}
		this.bytes = bytes;
		string size = Cgml.MiscUtils.printMemoryUse( bytes );
		statusMessage = $"Mistral model: {size}";
		this.dir = dir;
	}
}