namespace Mistral.Model;
using Cgml;

sealed class TemporaryTensors: TensorPool
{
	// === Per-layer temporaries ===

	// Attention temporaries
	public Tensor? norm;
	public Tensor? xq, xk, xv;
	public Tensor? scores;
	public Tensor? attnTemp1, attnTemp2, attnOut;
	public Tensor? attnKey, attnVal;

	// Feed Forward temporaries
	public Tensor? ff1, ff2;

	// === Global temporaries ===
	public Tensor? inpL;
	public Tensor? result;
	public Tensor? topPCounters, topP;
	public Tensor? logProbsTrimmed;
#if DEBUG
	public Tensor? dbgRowMajor;
#endif
}