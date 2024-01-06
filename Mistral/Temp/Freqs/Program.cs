namespace Freqs;
using System.Numerics;

internal class Program
{
	static void Main( string[] args )
	{
		TestPrecision.test(); return;
		TestV2.test();

		ReferenceData data = ReferenceData.load( @"C:\Temp\2remove\Mistral\Python\freqs_cis.tsv" );
		Generator gen = new Generator();

		int x = 44, y = 288;
		Complex r = data.lookup( x, y );
		Complex g = gen.generate( x, y );

		Console.WriteLine( "Lookup: {0}", r );
		Console.WriteLine( "Generate: {0}", g );
	}
}