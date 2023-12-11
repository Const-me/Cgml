namespace PackShaders;

enum eResourceKind: byte
{
	UAV,
	SRV,
	CBuffer,
}

sealed record class ResourceBinding
{
	public eResourceKind kind { get; init; }
	public byte slot { get; init; }
	public string name { get; init; }
	public string? comment { get; init; }
	public object? extraData { get; init; }
}

sealed record class ConstantBufferField
{
	public string name { get; init; }
	public byte idxVector { get; init; }
	public byte idxOffset { get; init; }
	public byte size { get; init; }
	public string csType { get; init; }
	public string? comment { get; init; }
}

sealed record class ShaderReflection
{
	public const string IgnoreMarker = "CODEGEN_IGNORE";

	public ResourceBinding[] bindings { get; init; }
	public string? comment { get; init; }
	public bool codegenIgnore => comment == IgnoreMarker;
}