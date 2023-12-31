#pragma warning disable CS0649 // Field is never assigned to, and will always have its default value
#pragma warning disable CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring the field as nullable.
namespace Mistral.Model;
using Cgml;
using System.Text.Json;
using System.Text.Json.Serialization;

sealed class ConfigJson
{
	[JsonInclude]
	public string[] architectures;
	[JsonInclude]
	public float attention_dropout;
	[JsonInclude]
	public int bos_token_id;
	[JsonInclude]
	public int eos_token_id;
	[JsonInclude]
	public string hidden_act;
	[JsonInclude]
	public int hidden_size;
	[JsonInclude]
	public float initializer_range;
	[JsonInclude]
	public int intermediate_size;
	[JsonInclude]
	public int max_position_embeddings;
	[JsonInclude]
	public string model_type;
	[JsonInclude]
	public int num_attention_heads;
	[JsonInclude]
	public int num_hidden_layers;
	[JsonInclude]
	public int num_key_value_heads;
	[JsonInclude]
	public float rms_norm_eps;
	[JsonInclude]
	public float rope_theta;
	[JsonInclude]
	public int? sliding_window;
	[JsonInclude]
	public bool tie_word_embeddings;
	[JsonInclude]
	public string torch_dtype;
	[JsonInclude]
	public string transformers_version;
	[JsonInclude]
	public bool use_cache;
	[JsonInclude]
	public int vocab_size;

	public static ParamsJson load( string jsonPath )
	{
		using var stream = File.OpenRead( jsonPath );
		ConfigJson config = JsonSerializer.Deserialize<ConfigJson>( stream ) ?? throw new ArgumentException();
		Logger.Debug( "Deserialized config.json" );

		ParamsJson res = new ParamsJson
		{
			dim = config.hidden_size,
			n_layers = config.num_hidden_layers,
			head_dim = config.hidden_size / config.num_attention_heads,
			hidden_dim = config.intermediate_size,
			n_heads = config.num_attention_heads,
			n_kv_heads = config.num_key_value_heads,
			norm_eps = config.rms_norm_eps,
			vocab_size = config.vocab_size,
			ropeTheta = config.rope_theta,
			modelVersion = eModelVersion.Instruct02,
		};
		res.sliding_window = config.sliding_window ?? 4096;
		return res;
	}
}