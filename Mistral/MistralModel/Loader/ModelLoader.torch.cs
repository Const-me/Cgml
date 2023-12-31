namespace Mistral;
using Cgml;
using Mistral.Model;
using System.Collections.Generic;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Torch;

public static partial class ModelLoader
{
	sealed class LoadTraits: Torch.LoadTraits
	{
		readonly eTensorLayout blockCompression;

		public LoadTraits( eTensorLayout codec = eTensorLayout.Dense )
		{
			blockCompression = codec;
		}

		readonly string[] compressedTensors = new string[]
		{
			// Version 0.1 names
			".feed_forward.w1.weight",
			".feed_forward.w2.weight",
			".feed_forward.w3.weight",
			".attention.wk.weight",
			".attention.wo.weight",
			".attention.wq.weight",
			".attention.wv.weight",

			// Version 0.2 names
			".mlp.up_proj.weight",
			".mlp.down_proj.weight",
			".mlp.gate_proj.weight",
			".self_attn.q_proj.weight",
			".self_attn.k_proj.weight",
			".self_attn.v_proj.weight",
			".self_attn.o_proj.weight",
		};

		public override eTensorLayout tensorVramLayout( string key )
		{
			foreach( string s in compressedTensors )
				if( key.EndsWith( s ) )
					return blockCompression;
			return eTensorLayout.Dense;
		}

		public override eLoadTransform tensorLoadTransform( eDataType storedType, string key ) =>
			eLoadTransform.Fp16MakeIeee;
	}

	static void dbgSummarizeTensors( IReadOnlyDictionary<string, iTensor> dict )
	{
		Vector128<long> mem = Vector128<long>.Zero;
		HashSet<eDataType> types = new HashSet<eDataType>();
		foreach( iTensor t in dict.Values )
		{
			sTensorDesc desc = t.getDesc();
			types.Add( desc.dataType );
			mem = Sse2.Add( mem, t.getMemoryUse() );
		}

		Logger.Debug( "Pytorch model summary: {0} tensors, memory usage {1}, data types [ {2} ]",
			dict.Count,
			MiscUtils.printMemoryUse( mem ),
			string.Join( ", ", types ) );
	}

	static void dbgSaveTensorList( IReadOnlyDictionary<string, iTensor> dict, string path )
	{
		string dir = Path.GetDirectoryName( path ) ?? throw new ArgumentException();
		Directory.CreateDirectory( dir );

		List<string> keys = dict.Keys.ToList();
		keys.Sort( StringComparer.InvariantCultureIgnoreCase );

		using var stream = File.CreateText( path );
		foreach( string key in keys )
		{
			iTensor tensor = dict[ key ];
			sTensorDesc desc = tensor.getDesc();
			stream.WriteLine( "{0}: {1}, {2}", key, desc.shape.description(), desc.dataType );
		}
	}

	/// <summary>Produce another dictionary with tensor key names translated from version 0.2 into the original version 0.1</summary>
	static Dictionary<string, iTensor> renameTensors02( IReadOnlyDictionary<string, iTensor> dict )
	{
		Dictionary<string, iTensor> result = new Dictionary<string, iTensor>( dict.Count );

		Dictionary<string, string> globals = new Dictionary<string, string>( 3 )
		{
			{ "embed_tokens.weight", "tok_embeddings.weight" },
			{ "lm_head.weight", "output.weight" },
			{ "norm.weight", "norm.weight" },
		};

		Dictionary<string, string> layer = new Dictionary<string, string>( 9 )
		{
			{ "input_layernorm.weight", "attention_norm.weight" },
			{ "post_attention_layernorm.weight", "ffn_norm.weight" },

			{ "self_attn.q_proj.weight", "attention.wq.weight" },
			{ "self_attn.k_proj.weight", "attention.wk.weight" },
			{ "self_attn.v_proj.weight", "attention.wv.weight" },
			{ "self_attn.o_proj.weight", "attention.wo.weight" },

			{ "mlp.gate_proj.weight", "feed_forward.w1.weight" },
			{ "mlp.up_proj.weight", "feed_forward.w3.weight" },
			{ "mlp.down_proj.weight", "feed_forward.w2.weight" },
		};

		foreach( var kvp in dict )
		{
			string key = kvp.Key;
			if( key.StartsWith( "model." ) )
				key = key.Substring( 6 );

			if( key.StartsWith( "layers." ) )
			{
				string[] fields = key.Split( '.', 3 );
				fields[ 2 ] = layer[ fields[ 2 ] ];
				key = string.Join( '.', fields );
			}
			else
				key = globals[ key ];

			result.Add( key, kvp.Value );
		}

		return result;
	}

	/// <summary>Import model in the original Python-targeted format</summary>
	public static iModel importTorch( TorchSource source, sDeviceParams deviceParams )
	{
		iModel impl( Device dev, in TorchSource source )
		{
			// Load vocabulary from the blob of bytes in that Google-defined weird format in tokenizer.model file on disk
			Tokenizer tokenizer;
			using( var stm = File.OpenRead( source.tokenizer ) )
				tokenizer = new Tokenizer( dev, stm, (int)stm.Length );

			try
			{
				TransformerIndex? index = TransformerIndex.tryLoad( source.weights );

				// Load the JSON; format and file name depends on the model version.
				ParamsJson json;
				string nameJson;
				if( null != index )
				{
					// Mistral-7B-Instruct-v0.2
					nameJson = "config.json";
					string pathJson = Path.Combine( source.weights, nameJson );
					if( !File.Exists( pathJson ) )
						throw new FileNotFoundException( "config.json is missing from the input directory", pathJson );
					json = ConfigJson.load( pathJson );
				}
				else
				{
					// Mistral-7B-instruct-v0.1
					nameJson = "params.json";
					string pathJson = Path.Combine( source.weights, nameJson );
					if( !File.Exists( pathJson ) )
						throw new FileNotFoundException( "params.json is missing from the input directory", pathJson );
					json = ParamsJson.load( pathJson );
				}

				if( json.vocab_size != tokenizer.vocabSize )
					throw new ArgumentException( $"{nameJson} and {Path.GetFileName( source.tokenizer )} disagree about vocabulary size" );

				using iWeightsLoader loader = TensorLoader.createLoader( dev.device, new LoadTraits( source.compression ) );

				if( null != index )
				{
					// Mistral-7B-Instruct-v0.2
					loader.loadTransformer( index );

					// dbgSaveTensorList( loader.tensors, @"C:\Temp\2remove\Mistral\tensors-02-in.txt" );
					Dictionary<string, iTensor> renamedTensors = renameTensors02( loader.tensors );
					// dbgSaveTensorList( renamedTensors, @"C:\Temp\2remove\Mistral\tensors-02-out.txt" );

					iModel result = new Model.Model( dev, json, tokenizer, renamedTensors );

					// Clear the dictionary stored in the loader, otherwise they all gonna be disposed
					// The model's constructor has cleared another dictionary, with renamed tensors.
					loader.tensors.Clear();

					return result;
				}
				else
				{
					// Mistral-7B-instruct-v0.1
					string[] files = Directory.GetFiles( source.weights, "consolidated.*.pth" );
					loader.loadMultipart( files );

					Dictionary<string, iTensor> tensors = loader.tensors;
					// dbgSummarizeTensors( tensors );
					// dbgSaveTensorList( tensors, @"C:\Temp\2remove\Mistral\tensors-01.txt" );
					return new Model.Model( dev, json, tokenizer, tensors );
				}
			}
			catch
			{
				tokenizer.Dispose();
				throw;
			}
		}
		return loadImpl( dev => impl( dev, source ), deviceParams );
	}
}