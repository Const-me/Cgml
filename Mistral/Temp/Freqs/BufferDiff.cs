namespace Freqs;

struct BufferDiff
{
	double maxAbsDiff, totalAbsDiff, sumSquares;
	int count;

	public void add( double a, double b )
	{
		double diff = a - b;
		sumSquares = Math.FusedMultiplyAdd( diff, diff, sumSquares );
		diff = Math.Abs( diff );
		maxAbsDiff = Math.Max( maxAbsDiff, diff );
		totalAbsDiff += diff;
		count++;
	}

	public override string ToString()
	{
		double avgAbsDiff = totalAbsDiff / count;
		double rms = Math.Sqrt( sumSquares / count );
		return $"maxAbsDiff {maxAbsDiff}, avgAbsDiff {avgAbsDiff}, RMS {rms}";
	}
}