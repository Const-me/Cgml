namespace Cgml;

/// <summary>Wrapper around a read-only span to access elements of a dense row major matrix</summary>
public ref struct MatrixView<T> where T : unmanaged
{
	internal readonly ReadOnlySpan<T> span;

	/// <summary>Size of the matrix</summary>
	public readonly uint2 size;

	/// <summary>Initialize the structure</summary>
	public MatrixView( ReadOnlySpan<T> span, uint2 size )
	{
		this.span = span;
		this.size = size;
		if( span.Length != size.x * size.y )
			throw new ArgumentException();
	}

	/// <summary>Make span index from 2D integer position of the element</summary>
	public int index( int x, int y )
	{
		if( x >= 0 && y >= 0 && x < size.x && y < size.y )
			return x + y * (int)size.x;
		throw new ArgumentOutOfRangeException();
	}

	/// <summary>Load matrix element using 2D integer position</summary>
	public T this[ int x, int y ] => span[ index( x, y ) ];

	/// <summary>Load matrix element using linear index in the span</summary>
	public T this[ int i ] => span[ i ];

	/// <summary>A string for debugger</summary>
	public override string ToString() => size.ToString();

	/// <summary>Span to read slice of a single row</summary>
	public ReadOnlySpan<T> rowSpan( int x, int y, int length )
	{
		int rsi = index( x, y );
		if( x + length <= size.x )
			return span.Slice( rsi, length );
		throw new ArgumentOutOfRangeException();
	}
}