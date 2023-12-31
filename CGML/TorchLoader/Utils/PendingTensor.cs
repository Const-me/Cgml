namespace Torch;
using Cgml;

readonly struct PendingTensor: IComparable<PendingTensor>
{
	public string key { get; init; }
	public Tensor tensor { get; init; }
	public override string ToString() => $"\"{key}\": {tensor}";

	public int payloadBytes =>
		tensor.shape.countElements() * tensor.storage.dataType.elementSize();

	public int CompareTo( PendingTensor other ) =>
		tensor.offset.CompareTo( other.tensor.offset );
}