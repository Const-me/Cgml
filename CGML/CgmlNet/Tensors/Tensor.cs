namespace Cgml;

/// <summary>A reusable tensor in VRAM with read+write usage</summary>
public sealed class Tensor: IDisposable
{
	iDevice device;
	iTensor? m_native = null;
	sTensorDesc m_desc = default;

	/// <summary>Initialize with a device</summary>
	public Tensor( iDevice device )
	{
		this.device = device;
	}

	/// <summary>Destroy the C++ object</summary>
	public void Dispose()
	{
		m_native?.Dispose();
		m_native = null;
	}

	/// <summary>Create a new tensor</summary>
	public iTensor create( sTensorDesc desc )
	{
		// if( desc.usage != eBufferUse.ReadWrite ) throw new ArgumentException();

		m_native = device.createTensor( ref desc, m_native );
		m_desc = desc;
		return m_native;
	}

	/// <summary>Create a new dense tensor</summary>
	public iTensor createDense( in Int128 size, eDataType dataType )
	{
		sTensorDesc desc = new sTensorDesc
		{
			shape = new TensorShape( size ),
			dataType = dataType,
			usage = eBufferUse.ReadWrite,
			layout = eTensorLayout.Dense
		};

		return create( desc );
	}

	/// <summary>Native object</summary>
	public iTensor native =>
		m_native ?? throw new ApplicationException( "Tensor was not created" );

	/// <summary>Size of the tensor</summary>
	public Int128 size => m_desc.shape.size;

	/// <summary>Strides of the tensor, expressed in elements</summary>
	public Int128 stride => m_desc.shape.stride;

	/// <summary>Shape of the tensor</summary>
	public TensorShape shape => m_desc.shape;

	/// <summary>Element type of the tensor</summary>
	public eDataType dataType => m_desc.dataType;

	/// <summary>A string for debugger</summary>
	public override string ToString()
	{
		if( null == m_native )
			return "<empty>";
		return $"{m_desc.shape}, {m_desc.dataType}";
	}

	/// <summary>Change tensor into another shape with the same count of elements, retaining the data</summary>
	public void view( in Int128 newSize )
	{
		if( null == m_native )
			throw new NullReferenceException();
		TensorShape newShape = new TensorShape( newSize );
		m_native.view( ref newShape );
		m_desc.shape = newShape;
	}

	/// <summary>Permute dimensions of the tensor</summary>
	/// <remarks>Allows to implement transpositions and similar view-only operations on tensors</remarks>
	public void permute( byte x, byte y, byte z, byte w )
	{
		if( null == m_native )
			throw new NullReferenceException();

		TensorShape newShape = m_desc.shape.permute( x, y, z, w );
		m_native.view( ref newShape );
		m_desc.shape = newShape;
	}

	/// <summary>Change tensor into another shape with the same count of elements, retaining the data</summary>
	public void view( int x, int y, int z, int w ) =>
		view( new Int128( x, y, z, w ) );

	/// <summary>Trim tensor to specified size</summary>
	/// <remarks>This method doesn't call GPU nor C++ code.<br/>
	/// It returns a new <c>Tensor</c> with the same native object and slightly different shape.</remarks>
	public Tensor trimmed( int x, int y, int z, int w )
	{
		if( m_desc.layout != eTensorLayout.Dense )
			throw new NotSupportedException();
		if( x <= 0 || y <= 0 || z <= 0 || w <= 0 || x > size.x || y > size.y || z > size.z || w > size.w )
			throw new ArgumentOutOfRangeException();

		Tensor res = new Tensor( device );
		res.m_native = m_native;
		res.m_desc = m_desc;
		res.m_desc.shape = new TensorShape( new Int128( x, y, z, w ), m_desc.shape.stride );
		return res;
	}
}