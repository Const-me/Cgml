namespace Freqs;
using System.Diagnostics;

static class TestV2
{
	const int dim = 128;
	const float ropeTheta = 1000000.0f;
	const int maxPositionEmbeddings = 32768;

	static double invFreqElement( int x )
	{
		const float minusHalfDimMul = -2.0f / dim;
		float exp = x * minusHalfDimMul;
		return Math.Pow( ropeTheta, exp );
	}

	static float[][] loadTsv( string tsv )
	{
		List<float[]> rows = new List<float[]>();
		using var file = File.OpenText( tsv );

		while( true )
		{
			string? line = file.ReadLine();
			if( null == line )
				break;
			if( string.IsNullOrWhiteSpace( line ) )
				continue;

			string[] fields = line.Split( '\t' );
			float[] arr = new float[ fields.Length ];
			for( int i = 0; i < fields.Length; i++ )
				arr[ i ] = float.Parse( fields[ i ] );
			rows.Add( arr );
		}
		return rows.ToArray();
	}

	static double embElement( int x, int y )
	{
		if( x < 0 || x >= dim )
			throw new ArgumentException();

		double invFreq = invFreqElement( x % ( dim / 2 ) );
		return invFreq * y;
	}

	public static void test()
	{
		float[][] cos = loadTsv( @"C:\Temp\2remove\Mistral02\pre-b0-rotaryCos.tsv" );
		float[][] sin = loadTsv( @"C:\Temp\2remove\Mistral02\pre-b0-rotarySin.tsv" );

		if( sin.Length != cos.Length )
			throw new ArgumentException();

		BufferDiff cosDiff = default;
		BufferDiff sinDiff = default;

		for( int y = 0; y < cos.Length; y++ )
		{
			for( int x = 0; x < dim; x++ )
			{
				double e = embElement( x, y );
				double c = Math.Cos( e );
				double s = Math.Sin( e );

				float cp = cos[ y ][ x ];
				float sp = sin[ y ][ x ];

				cosDiff.add( cp, c );
				sinDiff.add( sp, s );
			}
		}

		Console.WriteLine( "cos: {0}", cosDiff );
		Console.WriteLine( "sin: {0}", sinDiff );
	}

	public static void test0()
	{
		double[] invFreqPython = new double[]
		{
			1.0000e+00, 8.0584e-01, 6.4938e-01, 5.2330e-01, 4.2170e-01, 3.3982e-01,
			2.7384e-01, 2.2067e-01, 1.7783e-01, 1.4330e-01, 1.1548e-01, 9.3057e-02,
			7.4989e-02, 6.0430e-02, 4.8697e-02, 3.9242e-02, 3.1623e-02, 2.5483e-02,
			2.0535e-02, 1.6548e-02, 1.3335e-02, 1.0746e-02, 8.6596e-03, 6.9783e-03,
			5.6234e-03, 4.5316e-03, 3.6517e-03, 2.9427e-03, 2.3714e-03, 1.9110e-03,
			1.5399e-03, 1.2409e-03, 1.0000e-03, 8.0584e-04, 6.4938e-04, 5.2330e-04,
			4.2170e-04, 3.3982e-04, 2.7384e-04, 2.2067e-04, 1.7783e-04, 1.4330e-04,
			1.1548e-04, 9.3057e-05, 7.4989e-05, 6.0430e-05, 4.8697e-05, 3.9242e-05,
			3.1623e-05, 2.5483e-05, 2.0535e-05, 1.6548e-05, 1.3335e-05, 1.0746e-05,
			8.6596e-06, 6.9783e-06, 5.6234e-06, 4.5316e-06, 3.6517e-06, 2.9427e-06,
			2.3714e-06, 1.9110e-06, 1.5399e-06, 1.2409e-06
		};

		float[] invFreqCs = new float[ invFreqPython.Length ];
		for( int i = 0; i < invFreqCs.Length; i++ )
			invFreqCs[ i ] = (float)invFreqElement( i );

		Debugger.Break();
	}
}