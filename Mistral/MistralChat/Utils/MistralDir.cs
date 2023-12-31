namespace MistralChat;
using System.IO;
using Torch;

sealed class MistralDir
{
	public readonly string statusMessage;
	public readonly string? dir;
	public readonly long? bytes;

	public const string tokenizer = "tokenizer.model";
	readonly string[] requiredFiles01 =
		new string[] { "consolidate_info.json", "consolidated.00.pth", "params.json", tokenizer };
	readonly string[] requiredFiles02 =
		new string[] { TransformerIndex.indexFileName, "config.json", tokenizer };

	bool hasRequiredFiles01( string dir, out long totalBytes )
	{
		totalBytes = 0;
		long cb = 0;
		foreach( string req in requiredFiles01 )
		{
			string path = Path.Combine( dir, req );
			if( !File.Exists( path ) )
				return false;
			cb += new FileInfo( path ).Length;
		}
		totalBytes = cb;
		return true;
	}

	bool hasRequiredFiles02( string dir, out long totalBytes )
	{
		totalBytes = 0;
		long cb = 0;
		foreach( string req in requiredFiles02 )
		{
			string path = Path.Combine( dir, req );
			if( !File.Exists( path ) )
				return false;
			cb += new FileInfo( path ).Length;
		}

		TransformerIndex? index = TransformerIndex.tryLoad( dir );
		if( null == index )
			return false;
		long? dataSize = index.computeDataSize();
		if( dataSize.HasValue )
		{
			cb += dataSize.Value;
			totalBytes = cb;
			return true;
		}
		return false;
	}

	bool hasRequiredFiles( string dir, out long totalBytes )
	{
		if( hasRequiredFiles02( dir, out totalBytes ) )
			return true;
		if( hasRequiredFiles01( dir, out totalBytes ) )
			return true;
		return false;
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