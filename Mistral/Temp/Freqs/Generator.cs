namespace Freqs;
using System.Numerics;

record class Generator
{
	readonly int dim, end;
	readonly double halfDimMul;
	readonly double theta;

	public Generator( int dim = 128, int end = 128000, float theta = 10000.0f )
	{
		this.dim = dim;
		this.end = end;
		this.theta = theta;
		halfDimMul = -2.0 / dim;
	}

	public Complex generate( int x, int y )
	{
		if( x < 0 || x >= dim / 2 )
			throw new ArgumentOutOfRangeException( nameof( x ) );
		if( y < 0 || y >= end )
			throw new ArgumentOutOfRangeException( nameof( y ) );

		double freq = Math.Pow( theta, x * halfDimMul );
		double outer = freq * y;
		(double sin, double cos) = Math.SinCos( outer );
		return new Complex( cos, sin );
	}
}