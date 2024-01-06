#pragma warning disable CS0649 // Field is never assigned to, and will always have its default value
namespace Mistral.Model;
using Cgml;
using System.Text.Json;
using System.Text.Json.Serialization;

/// <summary><c>params.json</c> from Mistral-7B-instruct-v0.1 model</summary>
sealed class ParamsJson
{
	// Do not rename these fields, they must match the names used in the params.json configuration file

	[JsonInclude]
	public int dim; // 4096
	[JsonInclude]
	public int n_layers; // 32
	[JsonInclude]
	public int head_dim; // 128
	[JsonInclude]
	public int hidden_dim; // 14336
	[JsonInclude]
	public int n_heads; // 32
	[JsonInclude]
	public int n_kv_heads; // 8
	[JsonInclude]
	public float norm_eps; // 1e-05
	[JsonInclude]
	public int sliding_window; // 4096
	[JsonInclude]
	public int vocab_size; // 32000

	// The following parameters are only present in config.json of the newer version of the model
	[JsonIgnore]
	public float ropeTheta;
	[JsonIgnore]
	public eModelVersion modelVersion;

	public static ParamsJson load( Stream stream )
	{
		ParamsJson res = JsonSerializer.Deserialize<ParamsJson>( stream ) ?? throw new ArgumentException();
		res.ropeTheta = 10000.0f;
		res.modelVersion = eModelVersion.Original;
		Logger.Debug( "Deserialized params.json" );
		return res;
	}

	public static ParamsJson load( string jsonPath )
	{
		using var stream = File.OpenRead( jsonPath );
		return load( stream );
	}
}