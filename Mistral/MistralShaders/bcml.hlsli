// BCML codecs re-arranges compressed uint elements into column-major panels of the fixed height.
// Helps with performance because causes VRAM access to be fully coalesced when loading these elements.
// This constant defines the height of these panels
static const uint BCML_PANEL_HEIGHT = 64;

#ifndef USE_BF16
#define USE_BF16 0
#endif

// ==== Common functions and data structured for BCML1 and BCML2 codecs ====
#if BCML_CODEC == 1 || BCML_CODEC == 2

// Decoder state for BCML1 and BCML2 codecs
struct DecoderState
{
	// Bitmap with quantized weights.
	// When elements are consumed the bitmap is shifted to the right, i.e. next weight is in the lowest bits of this integer
	uint block;
	// Next index to load from the matrix
	uint rsi;
	// Multiplier + offset for the current block, from the header
	float2 header;
};

// Initial decoder state
DecoderState createDecoderState( uint rsiMatrix )
{
	DecoderState res;
	res.block = 0;
	res.rsi = rsiMatrix;
	res.header = 0.0;
	return res;
}

// Decode block header into 2 FP32 numbers; X is multiplier, Y is offset
inline float2 bcmlHeader( uint val )
{
#if USE_BF16
	uint2 fp32 = uint2( val << 16, val & 0xFFFF0000u );
	return asfloat( fp32 );
#else
	uint2 fp16 = uint2( val & 0xFFFF, val >> 16 );
	return f16tof32( fp16 );
#endif
}

// Count of elements in the complete compressed block of the matrix
static const uint BCML_BLOCK_SIZE = 32;
#endif

// ==== BCML1: 4 bits weights, 32 elements per block ====
#if BCML_CODEC == 1

// Decode BCML1 element from the lowest 4 bits of the integer
inline float bcml1Element( float2 header, uint val )
{
	float f = (float)(int)( val & 0xFu );
	return mad( f, header.x, header.y );
}

// Accumulate dot product of the rowBuffer slice [ j .. j + 31 ] with the next block of the compressed matrix
inline void computeCompressedBlock( inout uint j, inout DecoderState state, inout float acc )
{
	// Load block header from the matrix, and convert these FP16 numbers into float2 value
	state.header = bcmlHeader( mat.Load( state.rsi ) );
	state.rsi += 4 * BCML_PANEL_HEIGHT;

	for( uint i = 0; i < 4; i++ )
	{
		// Load 8 quantized weights from the matrix
		state.block = mat.Load( state.rsi );
		state.rsi += 4 * BCML_PANEL_HEIGHT;

		// Accumulate dot product of the rowBuffer slice [ j .. j + 7 ] with the corresponding 8 compressed elements of the matrix
		float m = bcml1Element( state.header, state.block );
		acc = mad( rowBuffer[ j ], m, acc );
		j++;

		// Funfact: DXBC assembly has a special shader instruction to extract bits like that:
		// https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/ubfe--sm5---asm-
		// Pretty similar to `bextr` CPU instruction from BMI1 set
		m = bcml1Element( state.header, state.block >> 4 );
		acc = mad( rowBuffer[ j ], m, acc );
		j++;

		m = bcml1Element( state.header, state.block >> 8 );
		acc = mad( rowBuffer[ j ], m, acc );
		j++;

		m = bcml1Element( state.header, state.block >> 12 );
		acc = mad( rowBuffer[ j ], m, acc );
		j++;

		m = bcml1Element( state.header, state.block >> 16 );
		acc = mad( rowBuffer[ j ], m, acc );
		j++;

		m = bcml1Element( state.header, state.block >> 20 );
		acc = mad( rowBuffer[ j ], m, acc );
		j++;

		m = bcml1Element( state.header, state.block >> 24 );
		acc = mad( rowBuffer[ j ], m, acc );
		j++;

		m = bcml1Element( state.header, state.block >> 28 );
		acc = mad( rowBuffer[ j ], m, acc );
		j++;
	}
}

// Handle 1 element of an incomplete block
inline void accumulateRemainderElement( uint j, inout DecoderState state, inout float acc )
{
	// According to disassembler, without these [branch] attributes compiler generates code with conditional moves
	// Which inflates count of loads by almost an order of magnitude

	[ branch ]
	if( 0 == ( j % 32 ) )
	{
		state.header = bcmlHeader( mat.Load( state.rsi ) );
		state.rsi += 4 * BCML_PANEL_HEIGHT;
	}

	[branch]
	if( 0 == ( j % 8 ) )
	{
		state.block = mat.Load( state.rsi );
		state.rsi += 4 * BCML_PANEL_HEIGHT;
	}

	// Decode current element of the matrix
	float m = bcml1Element( state.header, state.block );
	state.block >>= 4;

	// Update the accumulator
	float r = rowBuffer[ j ];
	acc = mad( r, m, acc );
}

// ==== BCML2: 3 bits weights, 32 elements per block ====
#elif BCML_CODEC == 2

// Decode BCML2 element from the lowest 3 bits of the integer
inline float bcml2Element( float2 header, uint val )
{
	float f = (float)(int)( val & 0x7u );
	return mad( f, header.x, header.y );
}

// Accumulate dot product of the rowBuffer slice [ j .. j + 31 ] with the next block of the compressed matrix
inline void computeCompressedBlock( inout uint j, inout DecoderState state, inout float acc )
{
	// Load block header from the matrix, and convert these FP16 numbers into float2 value
	state.header = bcmlHeader( mat.Load( state.rsi ) );
	state.rsi += 4 * BCML_PANEL_HEIGHT;

	// Load 32 bits from the matrix, they contain the first 10.66 quantized weights 
	state.block = mat.Load( state.rsi );
	state.rsi += 4 * BCML_PANEL_HEIGHT;

	// Accumulate dot product of the rowBuffer slice [ j .. j + 9 ] with the corresponding 10 compressed elements of the matrix
	float m = bcml2Element( state.header, state.block );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 3 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 6 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 9 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 12 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 15 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 18 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 21 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 24 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 27 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	// 10-th element of the compressed block uses bits [ 30 .. 32 ] of the payload
	// They are 2 bits from the previous uint value, 1 bit from the next one
	uint tmp = state.block >> 30;
	// Load next 32 bits from the matrix
	state.block = mat.Load( state.rsi );
	state.rsi += 4 * BCML_PANEL_HEIGHT;
	// Decode the element
	tmp |= ( state.block << 2 );
	m = bcml2Element( state.header, tmp );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 1 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 4 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 7 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 10 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 13 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 16 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 19 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 22 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 25 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 28 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	// 21-th element of the compressed block uses bits [ 63 .. 65 ] of the payload
	// That's 1 last bit from the previous uint value, 2 bits from the next one
	tmp = state.block >> 31;
	// Load next 32 bits from the matrix
	state.block = mat.Load( state.rsi );
	state.rsi += 4 * BCML_PANEL_HEIGHT;
	// Decode the element
	tmp |= ( state.block << 1 );
	m = bcml2Element( state.header, tmp );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 2 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 5 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 8 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 11 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 14 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 17 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 20 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 23 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 26 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;

	m = bcml2Element( state.header, state.block >> 29 );
	acc = mad( rowBuffer[ j ], m, acc );
	j++;
}

// Handle 1 element of an incomplete block
inline void accumulateRemainderElement( uint j, inout DecoderState state, inout float acc )
{
	const uint rem = j % 32;
	uint block;

	[branch]
	if( rem == 0 )
	{
		state.header = bcmlHeader( mat.Load( state.rsi ) );
		state.rsi += 4 * BCML_PANEL_HEIGHT;

		block = mat.Load( state.rsi );
		state.rsi += 4 * BCML_PANEL_HEIGHT;

		state.block = block >> 3;
	}
	else[ branch ] if( rem == 10 )
	{
		// 10-th element of the compressed block uses bits [ 30 .. 32 ] of the payload
		// They are 2 bits from the previous uint value, 1 bit from the next one
		const uint nextBlock = mat.Load( state.rsi );
		state.rsi += 4 * BCML_PANEL_HEIGHT;

		block = state.block | ( nextBlock << 2 );
		state.block = nextBlock >> 1;
	}
	else[ branch ] if( rem == 21 )
	{
		// 21-th element of the compressed block uses bits [ 63 .. 65 ] of the payload
		// That's 1 last bit from the previous uint value, 2 bits from the next one
		const uint nextBlock = mat.Load( state.rsi );
		state.rsi += 4 * BCML_PANEL_HEIGHT;

		block = state.block | ( nextBlock << 1 );
		state.block = nextBlock >> 2;
	}
	else
	{
		block = state.block;
		state.block >>= 3;
	}

	// Decode current element of the matrix
	float m = bcml2Element( state.header, block );

	// Update the accumulator
	float r = rowBuffer[ j ];
	acc = mad( r, m, acc );
}

// ==== BCML3 is actually reshaped GPTQ with 4 bits weights and 128 elements per block ====
#elif BCML_CODEC == 3

static const uint BCML_BLOCK_SIZE = 1024;

// Decoder state for BCML3 codec
struct DecoderState
{
	// Bitmap with quantized weights.
	// When elements are consumed the bitmap is shifted to the right, i.e. next weight is in the lowest bits of this integer
	uint block;
	// Next index to load from the matrix
	uint rsi;
	// Scale value
	float scale;
	// Bitmap from qzeros source tensor
	// Again, when inner blocks are consumed, the bitmap is shifted to the right
	uint zeros;
};

// Initial decoder state
DecoderState createDecoderState( uint rsiMatrix )
{
	DecoderState res;
	res.block = 0;
	res.rsi = rsiMatrix;
	res.scale = 0.0;
	res.zeros = 0;
	return res;
}

static const uint bcml3LoopsPerOuterBlock = 1024 / THREADS;

// The inner loop for the row vector*compressed matrix compute shader
inline void computeCompressedProduct( uint i, uint j, inout DecoderState state, inout float acc )
{
	[branch]
	if( 0 == ( j % 128 ) )
	{
		// Start of the inner block

		[branch]
		if( j == 0 && ( i % bcml3LoopsPerOuterBlock ) == 0 )
		{
			// Start of the outer block, load another zero value
			state.zeros = mat.Load( state.rsi );
			state.rsi += 4 * BCML_PANEL_HEIGHT;
		}
		else
		{
			// Advance to the next zero value, within the bitmap we already have
			state.zeros >>= 4;
		}

		// Load the scale
		state.scale = asfloat( mat.Load( state.rsi ) );
		state.rsi += 4 * BCML_PANEL_HEIGHT;
	}

	[branch]
	if( 0 == ( j % 8 ) )
	{
		// Need another bitmap with 8 weights, they are 4 bits/each for this codec
		state.block = mat.Load( state.rsi );
		state.rsi += 4 * BCML_PANEL_HEIGHT;
	}

	// Decode current element of the matrix
	int iWeight = (int)( state.block & 0xFu );
	state.block >>= 4;
	int iZero = (int)( state.zeros & 0xFu );
	float m = state.scale * (float)( iWeight - ( iZero + 1 ) );

	// Update the accumulator
	float r = rowBuffer[ j ];
	acc = mad( r, m, acc );
}

#else
#error Not Implemented
#endif

inline uint roundDownToBlock( uint i )
{
	// Round down by multiple of BCML_BLOCK_SIZE, using a bitwise trick
	const uint mask = ~( BCML_BLOCK_SIZE - 1 );
	return i & mask;
}