namespace Mistral.Model;
using Cgml;
using System.Diagnostics;
using System.Runtime.InteropServices;

partial struct Context
{
#if DEBUG
	static readonly string pathPythonDumps = @"C:\Temp\2remove\Mistral02";
	readonly string? dumps;
	public string? prefix;
#endif

#if DEBUG
	public void dbgCompareTensor( iTensor tensor, string zip )
	{
		if( null == dumps )
			return;
		TensorData data = context.downloadTensor( tensor );
		if( null != prefix )
			zip = $"{prefix}-{zip}";

		string path = Path.Combine( dumps, Path.ChangeExtension( zip, ".zip" ) );
		TensorData test = Torch.TensorLoader.load( path );
		TensorsDiff diff = data.diff( test );
		Logger.Debug( @"{0}: {1}, {2}", zip, data.desc.shape.description(), diff );
	}

	public void dbgCompareTensor( Tensor tensor, string zip ) =>
		dbgCompareTensor( tensor.native, zip );
#else
	public void dbgCompareTensor( iTensor tensor, string zip ) { }
	public void dbgCompareTensor( Tensor tensor, string zip ) { }
#endif

#if DEBUG
	readonly struct TopKTemp
	{
		public readonly float e;
		public readonly int idx;
		public TopKTemp( Half e, int idx )
		{
			this.e = (float)e;
			this.idx = idx;
		}
		public override string ToString() => $"{e} #{idx}";
	}

	static int topKCpu( Half[] sourceData, float rand, int topK )
	{
		TopKTemp[] temp = new TopKTemp[ sourceData.Length ];
		for( int i = 0; i < sourceData.Length; i++ )
			temp[ i ] = new TopKTemp( sourceData[ i ], i );
		// Sadly, Array.Sort is unstable sort, we need stable sorting here
		temp = temp.OrderByDescending( t => t.e ).ToArray();

		float threshold = temp[ topK - 1 ].e;
		while( topK < temp.Length && threshold == temp[ topK ].e )
			topK++;

		float maxVal = temp[ 0 ].e;
		double sumExp = 0;
		for( int i = 0; i < topK; i++ )
			sumExp += MathF.Exp( temp[ i ].e - maxVal );

		float mul = (float)( 1.0 / sumExp );
		float[] softmax = new float[ topK ];
		for( int i = 0; i < topK; i++ )
		{
			float f = temp[ i ].e;
			f = MathF.Exp( f - maxVal ) * mul;
			softmax[ i ] = f;
		}

		float sum = (float)softmax.Select( f => (double)f ).Sum();
		float scaledRand = rand * sum;

		float acc = 0;
		for( int i = topK - 1; i > 0; i-- )
		{
			acc += softmax[ i ];
			if( acc > scaledRand )
				return temp[ i ].idx;
		}
		return temp[ 0 ].idx;
	}

	public void testTopK()
	{
		const int length = 32000;
		const int topK = 50;

		// Generate source data in system memory
		// We don't want randomness while debugging stuff, seeding Random() with a constant
		Random rand = new Random( 0 );
		Half[] sourceData = new Half[ length ];
		for( int i = 0; i < length; i++ )
		{
			float r = rand.NextSingle();
			sourceData[ i ] = (Half)r;
		}

		// Upload to VRAM
		sTensorDesc desc = new sTensorDesc
		{
			shape = TensorShape.rowMajor( length ),
			dataType = eDataType.FP16,
			usage = eBufferUse.Immutable,
			layout = eTensorLayout.Dense,
		};
		using iTensor source = device.uploadImmutableTensor( desc, MemoryMarshal.AsBytes<Half>( sourceData ) );

		// Dispatch the shader
		Int128 size = desc.size;
		Int128 stride = desc.stride;
		if( size.yzw != new uint3( 1, 1, 1 ) || stride.x != 1 )
			throw new ArgumentException();
		Tensor temp = makeUintTensor( ref this.temp.topPCounters, 0x8000, 1 );
		Tensor res = makeUintTensor( ref this.temp.topP, 1, 1, eBufferUse.ReadWriteDownload );
		var cb = new ConstantBuffers.sampleTopK
		{
			width = (uint)size.x,
			topK = topK,
			rand = rand.NextSingle(),
		};
		context.sampleTopK( cb, temp.native, res.native, source );
		context.dispatch( 1 );

		// Download the result
		int resultGpu = -1;
		pfnReadTensor<int> pfnRead = delegate ( ReadOnlySpan<int> span )
		{
			Debug.Assert( span.Length == 1 );
			resultGpu = span[ 0 ];
		};
		context.download( res.native, pfnRead );

		// Compute reference version on CPU, using the same inputs including the random number
		int resultCpu = topKCpu( sourceData, cb.rand, topK );

		// Print these two integers
		Logger.Info( "testTopK: GPU {0}, CPU {1}", resultGpu, resultCpu );
		if( resultGpu != resultCpu )
			throw new ApplicationException( "Test failed" );
	}
#endif
}