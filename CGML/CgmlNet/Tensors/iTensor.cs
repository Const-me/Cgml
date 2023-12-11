namespace Cgml;
using ComLight;
using System.ComponentModel;
using System.Runtime.InteropServices;

/// <summary>A tensor in VRAM</summary>
[ComInterface( "30d2adce-78d3-4881-85e2-f7f00e898a2f", eMarshalDirection.ToManaged ), CustomConventions( typeof( Internal.NativeLogger ) )]
[DebuggerTypeProxy( typeof( TensorDebugView ) )]
public interface iTensor: IDisposable
{
	/// <summary>Get description structure of the tensor</summary>
	[EditorBrowsable( EditorBrowsableState.Never )]
	void getDesc( out sTensorDesc desc );

	/// <summary>Change tensor to another shape with the same count of elements, retaining the data</summary>
	void view( [In] ref TensorShape newShape );

	/// <summary>Get memory usage of the tensor; first ulong value is system RAM, second is VRAM.</summary>
	[EditorBrowsable( EditorBrowsableState.Never )]
	void getMemoryUse( out Int128 mem );
}