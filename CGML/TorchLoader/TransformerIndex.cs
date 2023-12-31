#pragma warning disable CS0649 // Field is never assigned to, and will always have its default value
#pragma warning disable CS8618 // Non-nullable property must contain a non-null value when exiting constructor
namespace Torch;
using System.Text.Json;
using System.Text.Json.Serialization;

/// <summary>Deserialized PyTorch index file <c>pytorch_model.bin.index.json</c></summary>
/// <seealso href="https://huggingface.co/docs/transformers/big_models" />
public sealed class TransformerIndex
{
	/// <summary>Name of the index file</summary>
	public const string indexFileName = "pytorch_model.bin.index.json";

	/// <summary>The metadata just consists of the total size of the model for now</summary>
	public sealed class Metadata
	{
		/// <summary>Total size of the model</summary>
		[JsonInclude]
		public long total_size;
	}

	/// <summary>The metadata</summary>
	[JsonInclude]
	public Metadata metadata;

	/// <summary>The main part of this index, which maps each parameter name to the file it's stored in</summary>
	[JsonInclude]
	public Dictionary<string, string> weight_map;

	[JsonIgnore]
	internal string directory { get; private set; }

	/// <summary>Try to deserialize index file stored in the directory</summary>
	public static TransformerIndex? tryLoad( string directory )
	{
		string jsonPath = Path.Combine( directory, indexFileName );
		if( !File.Exists( jsonPath ) )
			return null;

		using var stream = File.OpenRead( jsonPath );
		TransformerIndex? res = JsonSerializer.Deserialize<TransformerIndex>( stream );
		if( null == res )
			return null;
		res.directory = directory;
		return res;
	}

	/// <summary>Generate a sequence of absolute paths of all weight files mentioned in the index</summary>
	/// <remarks>The method doesn't check these files exist on disk</remarks>
	internal IEnumerable<string> listDataFiles()
	{
		HashSet<string> hashSet = new HashSet<string>( weight_map.Values );
		foreach( var name in hashSet )
			yield return Path.Combine( directory, name );
	}

	/// <summary>If the directory contains all data files listed in this index, return the combined size of these files.<br/>
	/// Otherwise, return <c>null</c>.</summary>
	public long? computeDataSize()
	{
		long cb = 0;
		foreach( string path in listDataFiles()  )
		{
			if( !File.Exists( path ) )
				return null;
			cb += new FileInfo( path ).Length;
		}
		return cb;
	}
}