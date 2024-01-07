namespace Mistral.Model;
using Cgml;
using Mistral;
using System.IO.Compression;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

sealed partial class Model: iModel
{
	string makeDescription()
	{
		long elts = transformer.totalWeights();
		double billion = (double)elts / 1.0E+9;

		long vramBytes = getMemoryUse().GetElement( 1 );
		string str = MiscUtils.printMemoryUse( vramBytes );
		eTensorLayout codec = transformer.compression;
		if( codec == eTensorLayout.Dense )
			return $"{billion:0.#} B weights, {str} VRAM";
		else
			return $"{billion:0.#} B weights, {str} VRAM, {codec}";
	}
	string iModel.description() => makeDescription();

	internal void printMemopyUse( bool stats )
	{
		if( stats )
		{
			Dictionary<string, Vector128<long>> dict = new Dictionary<string, Vector128<long>>();
			transformer.getMemoryUse( dict );
			dict.addTensorMemory( input );
			foreach( string line in dict.printVramStats() )
				Console.WriteLine( line );
		}
		else
			Logger.Info( "Loaded the model: {0}", makeDescription() );
	}

	Vector128<long> getMemoryUse()
	{
		Vector128<long> v = transformer.getMemoryUse();
		v = Sse2.Add( v, input.getMemoryUse() );
		return v;
	}

	IEnumerable<string> iModel.profilerMeasures() =>
		dev.context.profilerGetData().formatted();

	ProfilerData? iModel.profilerData()
	{
		ProfilerResult[]? arr = dev.context.profilerGetData();
		if( arr == null )
			return null;
		return new ProfilerData( arr );
	}

	eModelVersion iModel.modelVersion => transformer.modelVersion;

	const string vocabEntry = "tokenizer.model";

	public void save( string path, string vocab, Action<double>? pfnProgress )
	{
		Logger.Debug( "Saving CGML model.." );
		using var zipFile = File.Create( path );
		using var zip = new ZipArchive( zipFile, ZipArchiveMode.Create );

		zip.CreateEntryFromFile( vocab, vocabEntry, CompressionLevel.SmallestSize );

		Transformer.serializer().write( zip, transformer, dev.context, false, pfnProgress );

		Logger.Info( "Saved CGML model: {0}", path );
	}

	void iModel.getVideoMemoryUsage( out long kv, out long temp )
	{
		kv = 0;
		temp = input.getMemoryUse().GetElement( 1 );
		transformer.getVideoMemoryUsage( ref kv, ref temp );
	}

	readonly PerformanceParams performanceParams = new PerformanceParams();
	PerformanceParams iModel.performanceParams => performanceParams;
}