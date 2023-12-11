groupshared float reductionBuffer[ THREADS ];

// Compute horisontal sum of the numbers. The result is only correct on the thread #0 of the group.
inline void horizontalSum( const uint thread, inout float sum )
{
	reductionBuffer[ thread ] = sum;
	for( uint i = THREADS / 2; i > 1; i /= 2 )
	{
		GroupMemoryBarrierWithGroupSync();
		if( thread < i )
		{
			sum += reductionBuffer[ thread + i ];
			reductionBuffer[ thread ] = sum;
		}
	}
	GroupMemoryBarrierWithGroupSync();
	if( 0 == thread )
		sum += reductionBuffer[ 1 ];
}

// Compute horizontal maximum of the numbers. The result is only correct on the thread #0 of the group.
inline void horizontalMax( const uint thread, inout float val )
{
	reductionBuffer[ thread ] = val;
	for( uint i = THREADS / 2; i > 1; i /= 2 )
	{
		GroupMemoryBarrierWithGroupSync();
		if( thread < i )
		{
			val = max( val, reductionBuffer[ thread + i ] );
			reductionBuffer[ thread ] = val;
		}
	}
	GroupMemoryBarrierWithGroupSync();
	if( 0 == thread )
		val = max( val, reductionBuffer[ 1 ] );
}