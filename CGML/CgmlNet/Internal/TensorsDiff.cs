namespace Cgml;
using Cgml.Internal;

struct sTensorBuffer
{
	public sTensorDesc desc;
	public int lengthBytes;
}

/// <summary>Difference between two tensors</summary>
public struct TensorsDiff
{
	/// <summary>Maximum( abs( a - b ) )</summary>
	public readonly float maxAbsDiff;

	/// <summary>Average( abs( a - b ) )</summary>
	public readonly float avgAbsDiff;

	/// <summary>Root mean square of the differences</summary>
	/// <seealso href="https://en.wikipedia.org/wiki/Root_mean_square" />
	public readonly float rms;

	/// <summary>A string for debugger</summary>
	public override string ToString() =>
		$"maxAbsDiff {maxAbsDiff}, avgAbsDiff {avgAbsDiff}, RMS {rms}";

	/// <summary>Compare two tensors in system memory</summary>
	public static TensorsDiff compute( ReadOnlySpan<byte> a, in sTensorDesc ad, ReadOnlySpan<byte> b, in sTensorDesc bd )
	{
		sTensorBuffer atb = new sTensorBuffer
		{
			desc = ad,
			lengthBytes = a.Length
		};

		sTensorBuffer btb = new sTensorBuffer
		{
			desc = bd,
			lengthBytes = b.Length
		};

		TensorsDiff result;
		int hr;
		NativeLogger.prologue();
		unsafe
		{
			fixed( byte* p0 = a )
			fixed( byte* p1 = b )
			{
				hr = Library.dbgTensorsDiff( out result, (IntPtr)p0, ref atb, (IntPtr)p1, ref btb );
			}
		}
		NativeLogger.throwForHR( hr );
		return result;
	}
}