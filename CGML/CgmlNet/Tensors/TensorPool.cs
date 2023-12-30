namespace Cgml;
using System.Linq.Expressions;
using System.Reflection;
using System.Runtime.Intrinsics;

/// <summary>Abstract base class for a collection of temporary tensors</summary>
/// <remarks>It’s job is disposing all tensor fields of the derived class.<br/>
/// Derived classes are expected to have a bunch of fields of type <see cref="Tensor" />.</remarks>
public abstract class TensorPool: IDisposable
{
	/// <summary>List all tensors in this object</summary>
	protected IEnumerable<Tensor?> listTensors() => pfnListTensors( this );

	/// <summary>Release all tensors in the fields of the derived class</summary>
	public virtual void Dispose()
	{
		foreach( Tensor? t in listTensors() )
			t?.Dispose();
	}

	/// <summary>Compute VRAM memory used by all tensors in this object</summary>
	public long getVideoMemoryUsage()
	{
		long res = 0;
		foreach( Tensor? t in listTensors() )
			res += t.getMemoryUse().GetElement( 1 );
		return res;
	}

	readonly Func<object, IEnumerable<Tensor?>> pfnListTensors;

	/// <summary>Use reflection + runtime code generation to build the <c>pfnListTensors</c> function</summary>
	public TensorPool()
	{
		pfnListTensors = lookup( GetType() );
	}

	static readonly object syncRoot = new object();
	static readonly Dictionary<Type, Func<object, IEnumerable<Tensor?>>> typeCache = new Dictionary<Type, Func<object, IEnumerable<Tensor?>>>();

	static Func<object, IEnumerable<Tensor?>> lookup( Type t )
	{
		lock( syncRoot )
		{
			if( typeCache.TryGetValue( t, out var pfn ) )
				return pfn;
			pfn = build( t );
			typeCache.Add( t, pfn );
			return pfn;
		}
	}

	static Func<object, IEnumerable<Tensor?>> build( Type t )
	{
		ParameterExpression[] objParam = new ParameterExpression[ 1 ] { Expression.Parameter( typeof( object ), "obj" ) };
		ParameterExpression arr = Expression.Parameter( typeof( Tensor[] ), "arr" );
		LabelTarget returnTarget = Expression.Label( typeof( IEnumerable<Tensor?> ) );

		FieldInfo[] fields = t.GetFields( BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic );
		fields = fields.Where( f => f.FieldType == typeof( Tensor ) ).ToArray();

		ParameterExpression localvarTypedParam = Expression.Parameter( t, "objTyped" );
		List<Expression> block = new List<Expression>( fields.Length + 4 );
		var newArray = Expression.NewArrayBounds( typeof( Tensor ), Expression.Constant( fields.Length ) );
		block.Add( Expression.Assign( localvarTypedParam, Expression.Convert( objParam[ 0 ], t ) ) );
		block.Add( Expression.Assign( arr, newArray ) );

		for( int i = 0; i < fields.Length; i++ )
		{
			var lhs = Expression.ArrayAccess( arr, Expression.Constant( i ) );
			var rhs = Expression.Field( localvarTypedParam, fields[ i ] );
			block.Add( Expression.Assign( lhs, rhs ) );
		}
		block.Add( Expression.Return( returnTarget, arr ) );
		block.Add( Expression.Label( returnTarget, Expression.Constant( null, typeof( IEnumerable<Tensor?> ) ) ) );

		ParameterExpression[] localVars = new ParameterExpression[ 2 ]{ localvarTypedParam, arr };
		var blockExpr = Expression.Block( localVars, block );
		var lambda = Expression.Lambda<Func<object, IEnumerable<Tensor?>>>( blockExpr, objParam );
		var result = lambda.Compile();
		return result;
	}
}