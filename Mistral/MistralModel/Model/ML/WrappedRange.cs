namespace Mistral.Model;
using System.Diagnostics;

/// <summary>Rotation of a tensor along some coordinate</summary>
readonly struct WrappedRange
{
	/// <summary>Start offset of the first slice in the range</summary>
	public readonly int offset0;
	/// <summary>Length of the first slice in the range</summary>
	public readonly int length0;
	/// <summary>Start offset of the second slice in the range</summary>
	public readonly int offset1;
	/// <summary>Length of the second slice in the range</summary>
	public readonly int length1;

	/// <summary>Length of the range</summary>
	public int length => length0 + length1;

	WrappedRange( int off0, int len0, int off1, int len1 )
	{
		offset0 = off0;
		length0 = len0;
		offset1 = off1;
		length1 = len1;
	}

	public static WrappedRange single( int length, int start = 0 )
	{
		if( length <= 0 || start < 0 )
			throw new ArgumentOutOfRangeException();
		return new WrappedRange( start, length, 0, 0 );
	}

	public static WrappedRange wrapped( int startPosition, int wrapWidth, int length, bool trimToEnd )
	{
		if( startPosition < 0 || startPosition >= wrapWidth || wrapWidth <= 0 || length <= 0 )
			throw new ArgumentOutOfRangeException();

		if( length > wrapWidth )
		{
			// Cutting a slice which is longer that the source tensor
			// We don't have that much data, decrement the length
			if( trimToEnd )
			{
				startPosition += ( length - wrapWidth );
				startPosition %= wrapWidth;
			}
			length = wrapWidth;
		}

		int unwrappedEnd = startPosition + length;
		if( unwrappedEnd <= wrapWidth )
		{
			// No wrapping required, produce a single slice
			return new WrappedRange( startPosition, length, 0, 0 );
		}
		else
		{
			// The first slice is [ startPosition .. wrapWidth ), the second slice starts from 0
			int length0 = wrapWidth - startPosition;
			int length1 = length - length0;
			Debug.Assert( length0 > 0 && length1 > 0 );
			return new WrappedRange( startPosition, length0, 0, length1 );
		}
	}

	/// <summary>True when this structure contains two slices</summary>
	public bool isWrapped => 0 != length1;

	/// <summary>Integer to add to the output position in the second slice to find input position.</summary>
	public int inputOffset1 => offset1 - length0;

	public int endOffset => isWrapped ? Math.Max( offset0 + length0, offset1 + length1 ) : offset0 + length0;
}