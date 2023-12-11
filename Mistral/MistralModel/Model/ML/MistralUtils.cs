namespace Mistral.Model;
using Cgml;
using System.Diagnostics;

static class MistralUtils
{
	/// <summary>Fill the complete tensor with zero elements</summary>
	public static void writeDenseZeros( this iContext context, iTensor tensor, in Int128 size )
	{
		int length = size.horizontalProduct();
		const int VALS_PER_GROUP = 0x10000;
		int groups = ( length + ( VALS_PER_GROUP - 1 ) ) / VALS_PER_GROUP;

		var cb = new ConstantBuffers.memsetFloat
		{
			bufferLength = (uint)length,
		};
		context.memsetFloat( cb, tensor );
		context.dispatch( groups );
	}

	// Split [ 0 .. total - 1 ] interval into slices, so that every slice does not exceed maxBatch
	// Generate a sequence of [ startOffset, sliceLength ] pairs
	public static IEnumerable<(int, int)> batchSplit( int total, int maxBatch )
	{
		Debug.Assert( maxBatch > 0 );
		if( total > maxBatch )
		{
			if( 0 != ( total % maxBatch ) )
			{
				int count = ( total + maxBatch - 1 ) / maxBatch;
				int min = total / count;
				int rem = total % count;

				int off = 0;
				for( int i = 0; i < count; i++ )
				{
					int len = min;
					len += ( i < rem ) ? 1 : 0;
					yield return (off, len);
					off += len;
				}
			}
			else
			{
				// They divide evenly
				int count = total / maxBatch;
				int off = 0;
				for( int i = 0; i < count; i++, off += maxBatch )
					yield return (off, maxBatch);
			}
		}
		else
		{
			// The interval fits in a single batch
			yield return (0, total);
		}
	}
}