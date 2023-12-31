namespace Mistral;
using Cgml;

static class Program
{
	static string? gpu => null;

	const string pathTokenizer = @"D:\Data\Mistral\Mistral-7B-Instruct-v0.2\tokenizer.model";
	static iModel loadOrig()
	{
		var source = new TorchSource()
		{
			tokenizer = pathTokenizer,
			// weights = @"D:\Data\Mistral\mistral-7B-v0.1",
			// weights = @"D:\Data\Mistral\Mistral-7B-instruct-v0.1",
			weights = @"D:\Data\Mistral\Mistral-7B-Instruct-v0.2",
			compression = eTensorLayout.BCML1,
		};
		return ModelLoader.importTorch( source, new sDeviceParams( gpu, null ) );
	}

	// const string pathCompressed = @"D:\Data\Mistral\mistral-7B-instruct.cgml";
	const string pathCompressed = @"D:\Data\Mistral\Mistral-7B-instruct02\Mistral-7B-Instruct-02.cgml";
	static void saveCgml( iModel model )
	{
		ModelLoader.save( model, pathTokenizer, pathCompressed );
	}

	static iModel loadCompressed()
	{
		return ModelLoader.load( pathCompressed, new sDeviceParams( gpu, null ) );
	}

	static void convert()
	{
		using iModel model = loadOrig(); saveCgml( model );
	}

	static void mainImpl( string[] args )
	{
		ConsoleLogger.setup( eLogLevel.Debug, eLoggerFlags.SkipFormatMessage );
		// convert(); return;
		using iModel model = loadCompressed();
		// using iModel model = loadOrig();

		// Prompt
		string prompts = "I believe the meaning of life is";

		// Generate and print the results
		const int maxTokens = 35;
		{
			string result = model.generate( prompts, maxTokens );
			Console.WriteLine( result );
			Console.WriteLine( "=====================" );
		}

		// Also print shader profiler data
		foreach( string p in model.profilerMeasures() )
			Console.WriteLine( p );
	}

	static void Main( string[] args )
	{
		try
		{
			mainImpl( args );
		}
		catch( Exception e )
		{
			Console.WriteLine( e.ToString() );
		}
	}
}