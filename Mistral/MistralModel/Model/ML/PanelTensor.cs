namespace Mistral.Model;
using Cgml;

/// <summary>A tensor reshaped into 32 elements tall column major panels</summary>
readonly struct PanelTensor
{
	public readonly Tensor dense;
	public readonly TensorShape shape;
	public const int PANEL_HEIGHT = 32;

	public PanelTensor( Tensor dense, Int128 size )
	{
		this.dense = dense;
		Int128 stride = new Int128( 0, size.x * PANEL_HEIGHT, dense.stride.z, dense.stride.w );
		shape = new TensorShape( size, stride );
	}

	public Int128 size => shape.size;
	public Int128 stride => shape.stride;
	public iTensor native => dense.native;

	public override string ToString() => shape.ToString();
}