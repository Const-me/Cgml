#pragma once
#include "../API/iTensor.cl.h"

namespace Cgml
{
	// Extract ID3D11UnorderedAccessView stored in the tensor object
	// For immutable tensors, the function silently returns nullptr.
	ID3D11UnorderedAccessView* getTensorUav( iTensor* tensor );
}