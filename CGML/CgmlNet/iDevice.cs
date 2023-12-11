namespace Cgml;
using ComLight;
using System.ComponentModel;
using System.Runtime.InteropServices;

/// <summary>The device interface represents a virtual adapter; it is used to create resources.</summary>
[ComInterface( "a90d9b4f-be31-495c-abbb-a7b1d1d58c57", eMarshalDirection.ToManaged ), CustomConventions( typeof( Internal.NativeLogger ) )]
public interface iDevice: IDisposable
{
	/// <summary>Create a new tensor in VRAM</summary>
	[RetValIndex]
	iTensor createTensor( [In] ref sTensorDesc desc, iTensor? reuse = null );

	/// <summary>Create immutable tensor in VRAM from the data in system memory</summary>
	[RetValIndex, EditorBrowsable( EditorBrowsableState.Never )]
	iTensor uploadImmutableTensor( [In] ref sTensorDesc desc, IntPtr rsi, int length );

	/// <summary>Create a tensor, but don’t create any GPU resources</summary>
	[RetValIndex]
	iTensor createUninitializedTensor( [In] ref sTensorDesc desc );

	/// <summary>Upload tensor data to VRAM</summary>
	/// <param name="tensor">An unitialized tensor made with <see cref="createUninitializedTensor" /> method</param>
	/// <param name="source">Stream with the payload data of the tensor</param>
	/// <param name="length">Count of bytes to read form the stream</param>
	void loadTensor( iTensor tensor, [ReadStream] Stream source, int length );

	/// <summary>Create immutable tensor in VRAM from the supplied stream</summary>
	/// <remarks>The input stream needs to be be dense.<br/>
	/// If you pass <see cref="eTensorLayout.BCML1" /> option, this method will reshape tensor on CPU before uploading.</remarks>
	[RetValIndex]
	iTensor loadImmutableTensor( [In] ref sTensorDesc desc, [ReadStream] Stream source, int length, eLoadTransform tform = eLoadTransform.None );

	/// <summary>loadImmutableTensor with block compression uses a thread pool.<br/>
	/// This method waits for these compression jobs, if any</summary>
	void waitForWeightsCompressor();

	/// <summary>Information about the D3D device</summary>
	[RetValIndex]
	sDeviceInfo getDeviceInfo();

	/// <summary>Load the vocabulary</summary>
	/// <remarks>This method doesn’t touch the GPU.<br />
	/// The reason why it’s here in this COM interface, we want custom marshaling for the stream.
	/// That feature is implemented in ComLight runtime, not gonna work for a DLL imported function.</remarks>
	[RetValIndex]
	SentencePiece.iProcessor loadSentencePieceModel( [ReadStream] Stream source, int length );
}