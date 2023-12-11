namespace Torch;
using System.Diagnostics;
using System.Security.Cryptography;

static class MiscUtils
{
	/// <summary>True when the sequence is made of equal elements</summary>
	public static bool same<E>( this IEnumerable<E> list )
	{
		var cmp = EqualityComparer<E>.Default;
		bool first = true;
		E? firstElement = default;
		foreach( var i in list )
		{
			if( first )
			{
				first = false;
				firstElement = i;
			}
			else if( !cmp.Equals( firstElement, i ) )
				return false;
		}
		return true;
	}

	public static Guid md5( ReadOnlySpan<byte> bytes )
	{
		using MD5 md5 = MD5.Create();
		byte[] hash = md5.ComputeHash( bytes.ToArray() );
		return new Guid( hash );
	}

#if !NET7_0_OR_GREATER
	/// <summary>Compatibility for .NET &lt; 7.0</summary>
	public static void ReadExactly( this Stream stream, Span<byte> buffer )
	{
		while( !buffer.IsEmpty )
		{
			int received = stream.Read( buffer );
			if( received <= 0 )
				throw new EndOfStreamException();
			Debug.Assert( received <= buffer.Length );
			buffer = buffer.Slice( received );
		}
	}
#endif
}