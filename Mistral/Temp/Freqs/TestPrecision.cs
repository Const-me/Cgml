namespace Freqs;
using System.Numerics;

static class TestPrecision
{
	const int dim = 128;
	const float minusHalfDimMul = -2.0f / dim;
	const float ropeTheta = 1000000.0f;

	/// <summary>High-precision FP64 version unavailable on GPUs</summary>
	/// <remarks>GPUs don’t have FP64 version of <c>sincos</c> intrinsic</remarks>
	static Vector2 fp64( float x, int y )
	{
		double angle = (double)x * y;
		(double sin, double cos) = Math.SinCos( angle );
		return new Vector2( (float)cos, (float)sin );
	}

	/// <summary><c>[ cos( a ), sin( a ) ]</c> computed in FP32 precision</summary>
	static Vector2 sincos( float a )
	{
		(float sin, float cos) = MathF.SinCos( a );
		return new Vector2( cos, sin );
	}

	/// <summary>FP32 version which replicates the math of <c>rotaryEmbedding.hlsl</c></summary>
	static Vector2 fp32Orig( float x, int y )
	{
		float angle = x * y;
		return sincos( angle );
	}

	/// <summary>Mixed-precision version which replicates the math of <c>rotaryEmbedding.fp[12].hlsl</c>,<br/>
	/// based on the third parameter</summary>
	static Vector2 hlsl64( float x, int y, bool fp2 )
	{
		const double twoPi = Math.PI * 2;
		const double invTwoPi = 1.0 / twoPi;

		double angle;
		if( fp2 )
		{
			angle = (double)x * y;
			double div = angle * invTwoPi;
			double i = (int)div;
			angle = Math.FusedMultiplyAdd( i, -twoPi, angle );
		}
		else
		{
			angle = (double)x * (float)y;
			double div = angle * invTwoPi;
			float i = MathF.Floor( (float)div );
			angle -= twoPi * i;
		}

		return sincos( (float)angle );
	}

	public static void test()
	{
		float x = 0;
		x = MathF.Pow( ropeTheta, x * minusHalfDimMul );	// 1.0
		int y = 31999;

		// wolframalpha.com:
		// cos( 31999 ) = 0.3031789574696147056581569467402158961570311406306820473980907898...
		// sin( 31999 ) = -0.952933638690353693336759806466130436818556821144376796602382312...
		Console.WriteLine( "fp64: {0}", fp64( x, y ) );
		Console.WriteLine( "gpu.0: {0}", fp32Orig( x, y ) );
		Console.WriteLine( "gpu.1: {0}", hlsl64( x, y, false ) );
		Console.WriteLine( "gpu.2: {0}", hlsl64( x, y, true ) );
	}
}