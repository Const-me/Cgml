#if USE_FP64
// Wrap a potentialy large FP64 value around 2*pi; the result is then passed to sincos() intrinsic
inline float wrapTwoPi( double x )
{
	const double twoPi = 6.283185307179586476925286766559;
	const double invTwoPi = 0.15915494309189533576888376337251;
	double div = x * invTwoPi;
	
#if USE_FP64 > 1
	// GPU supports FP64 FMA instruction, also itod / dtoi to convert numbers between int32 / fp64
	int i = (int)div;
	x = fma( (double)i, -twoPi, x );
#else
	// GPU doesn't support FMA, but it does support add/sub/mul
	float i = floor((float)div);
	x -= twoPi * i;
#endif
	return (float)x;
}
#endif

// Generate values which replace freqs_cis pre-computed tensor which is used in the original Python version
// Or sin/cos pre-computed tensors in MistralRotaryEmbedding class from transformers-4.36.2/src/transformers/models/mistral/modeling_mistral.py
const float2 computeFreqs( int2 i2 )
{
	// Not sure the precision is good enough for the use case. The original code uses FP64 precision for these things
	// Might need another version of this shaders, for GPUs which support FP64 math
	// OTOH, the output is then downcasted to FP16. It's also possible Facebook used FP64 math by mistake.
	float2 f2 = (float2) i2;

	// Disable the warning X3571 "pow(f, e) will not work for negative f"
	// The theta is from the constant buffer, and CPU-running code sets that value to a large positive number like 10000.0
#pragma warning( disable : 3571 )
	float freq = pow( theta, f2.x * minusHalfDimMul );
#if USE_FP64
	double tmp = freq;
#if USE_FP64 > 1
	tmp *= (double)i2.y;
#else
	// Microsoft neglected to documented that, but unfortunately `itod` instruction is an extended fp64 op.
	// Let's hope total count of these tokens won't exceed 16777216
	tmp *= f2.y;
#endif
	float outer = wrapTwoPi( tmp );
#else
	float outer = freq * f2.y;
#endif

	float2 result;
	sincos( outer, result.y, result.x );
	return result;
}