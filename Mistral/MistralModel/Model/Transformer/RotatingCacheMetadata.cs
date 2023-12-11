namespace Mistral.Model;
using Cgml;

/// <summary>Readonly interface to get state of the circular KV caches</summary>
interface iRotatingCacheMetadata
{
	/// <summary>Absolute position used for rope</summary>
	int absolute { get; }

	/// <summary>Wrapped position to update the cached tensors</summary>
	int storePosition { get; }

	/// <summary>Make unrotated view of the cache tensor</summary>
	RotatedTensorShape loadShape( Tensor cache );
}

/// <summary>Utility class to track state of the circular KV caches</summary>
sealed class RotatingCacheMetadata: iRotatingCacheMetadata
{
	readonly int window;
	public RotatingCacheMetadata( int window )
	{
		this.window = window;
		absolute = 0;
		storePosition = 0;
		loadRange = default;
	}

	/// <summary>The constructor used by <see cref="iModel.stateRestore(iModelState?)" /> method</summary>
	public RotatingCacheMetadata( int window, int absolute )
	{
		this.window = window;
		this.absolute = absolute;
		storePosition = absolute % window;
		loadRange = default;
	}

	/// <summary>Call this before running the transformer, pass count of input tokens</summary>
	public void begin( int countTokens = 1 )
	{
		if( countTokens < 1 )
			throw new ArgumentOutOfRangeException( nameof( countTokens ) );

		int end = absolute + countTokens;
		if( end <= window )
		{
			// Not enough cached data so these circulars tensors ain't wrapped yet.
			// Load from the initial slice of the cache
			loadRange = WrappedRange.single( end );
		}
		else
		{
			// Need to unwrap the circular buffer while loading from these caches
			loadRange = WrappedRange.wrapped( ( end - window ) % window, window, window, false );
		}
	}

	/// <summary>Call this after running the transformer</summary>
	/// <param name="countTokens">The same integer which was passed into <see cref="begin(int)" /> method.</param>
	public void end( int countTokens = 1 )
	{
		if( countTokens < 1 )
			throw new ArgumentOutOfRangeException( nameof( countTokens ) );

		absolute += countTokens;
		storePosition = absolute % window;
	}

	/// <summary>Absolute position used for rope</summary>
	public int absolute { get; private set; }

	/// <summary>Wrapped position to update the cached tensors</summary>
	public int storePosition { get; private set; }

	/// <summary>Range to load from the cached tensors</summary>
	WrappedRange loadRange;

	RotatedTensorShape iRotatingCacheMetadata.loadShape( Tensor cache )
	{
		RotatedTensorShape res = RotatedTensorShape.createDense( cache.shape );
		res.unwrapZ( loadRange );
		return res;
	}
}