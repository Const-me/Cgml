namespace Mistral;
using Cgml;
using Mistral.Model;
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
			".feed_forward.w1.weight",
			".feed_forward.w2.weight",
			".feed_forward.w3.weight",
			".attention.wk.weight",
			".attention.wo.weight",
			".attention.wq.weight",
			".attention.wv.weight"
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

	/// <summary>Import model in the original Python-targeted format</summary>
	public static iModel importTorch( TorchSource source, sDeviceParams deviceParams )
	{
		iModel impl( Device dev, in TorchSource source )
		{
			// Load the JSON
			string pathJson = Path.Combine( source.weights, "params.json" );
			if( !File.Exists( pathJson ) )
				throw new FileNotFoundException( "params.json is missing from the input directory", pathJson );
			ParamsJson json = ParamsJson.load( pathJson );

			// Load vocabulary from the blob of bytes in that Google-defined weird format in tokenizer.model file on disk
			Tokenizer tokenizer;
			using( var stm = File.OpenRead( source.tokenizer ) )
				tokenizer = new Tokenizer( dev, stm, (int)stm.Length );

			try
			{
				if( json.vocab_size != tokenizer.vocabSize )
					throw new ArgumentException( $"params.json and {Path.GetFileName( source.tokenizer )} disagree about vocabulary size" );

				using var loader = TensorLoader.createLoader( dev.device, new LoadTraits( source.compression ) );
				string[] files = Directory.GetFiles( source.weights, "consolidated.*.pth" );
				loader.loadMultipart( files );

				Dictionary<string, iTensor> tensors = loader.tensors;
				// Apparently, all ~7.2E+9 numbers in the model are in BF16 format
				// dbgSummarizeTensors( tensors );
				return new Model.Model( dev, json, tokenizer, tensors );
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