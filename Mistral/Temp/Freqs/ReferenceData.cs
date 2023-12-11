namespace Freqs;
using System.Numerics;
using System.Text.RegularExpressions;

readonly struct ReferenceData
{
	readonly Complex[][] matrix;
	ReferenceData( Complex[][] matrix ) => this.matrix = matrix;

	// BTW, the C# parser is by ChatGPT

	// Define a regular expression to match the complex number pattern
	static readonly Regex regex = new Regex( @"\((?<real>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?<imaginary>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)j\)" );

	static Complex parseComplex( string complexString )
	{
		// Match the regular expression against the input string
		Match match = regex.Match( complexString );

		if( !match.Success )
			throw new FormatException( "Invalid complex number format" );
		// Extract real and imaginary parts from the named groups in the regex
		string realPart = match.Groups[ "real" ].Value;
		string imaginaryPart = match.Groups[ "imaginary" ].Value;

		// Parse the string parts into double values
		double real = double.Parse( realPart );
		double imaginary = double.Parse( imaginaryPart );

		// Create and return a Complex number
		return new Complex( real, imaginary );
	}

	public static ReferenceData load( string tsv )
	{
		List<Complex[]> rows = new List<Complex[]>();
		using var file = File.OpenText( tsv );
		while( true )
		{
			string? line = file.ReadLine();
			if( null == line )
				break;
			if( string.IsNullOrWhiteSpace( line ) )
				continue;

			string[] fields = line.Split( '\t' );
			Complex[] arr = new Complex[ fields.Length ];
			for( int i = 0; i < fields.Length; i++ )
				arr[ i ] = parseComplex( fields[ i ] );
			rows.Add( arr );
		}

		return new ReferenceData( rows.ToArray() );
	}

	public Complex lookup( int x, int y ) =>
		matrix[ y ][ x ];
}