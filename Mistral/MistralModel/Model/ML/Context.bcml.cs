namespace Mistral.Model;
using Cgml;
using System.Diagnostics;
#pragma warning disable CS0162 // Unreachable code detected, due to bfloat16 constant                                                        

readonly partial struct Context
{
	struct RowMatProductCb
	{
		public int rowLength;
		public int rowsCount;
		public uint2 arg0Strides;
		public uint2 resultStrides;
		public int matrixStride;
	}

	void columnProductCompressed( Tensor res, Tensor a, iTensor b, in sTensorDesc bDesc )
	{
		Debug.Assert( bDesc.stride.x == 0 );

		var cb = new RowMatProductCb
		{
			rowLength = a.size.x,
			rowsCount = res.size.x,
			arg0Strides = new uint2( a.stride.y, a.stride.z ),
			resultStrides = new uint2( res.stride.y, res.stride.z ),
			matrixStride = bDesc.stride.y,
		};

		const int THREADS = 256;

		switch( bDesc.layout )
		{
			case eTensorLayout.BCML1:
				if( bfloat16 )
					throw new ArgumentException();
				context.bindShader( (ushort)eShader.rowMatProductBc1, ref cb );
				context.bindTensors2( res.native, a.native, b );
				break;
			/*
			case eTensorLayout.BCML1E:
				if( !bfloat16 )
					throw new ArgumentException();
				context.bindShader( (ushort)eShader.rowMatProductBc1, ref cb );
				context.bindTensors2( res.native, a.native, b );
				break;
			case eTensorLayout.BCML2:
				if( bfloat16 )
					throw new ArgumentException();
				context.bindShader( (ushort)eShader.rowMatProductBc2, ref cb );
				context.bindTensors2( res.native, a.native, b );
				break;
			case eTensorLayout.BCML3:
				context.bindShader( (ushort)eShader.rowMatProductBc3, ref cb );
				context.bindTensors2( res.native, a.native, b );
				break;
			case eTensorLayout.BCML4:
				context.bindShader( (ushort)eShader.rowMatProductBc4, ref cb );
				context.bindTensors2( res.native, a.native, b );
				break;
			*/
			default:
				throw new NotImplementedException();
		}

		int groupsX = ( res.size.x + THREADS - 1 ) / THREADS;
		context.dispatch( groupsX, a.size.y, a.size.z );
	}
}