#include <stdafx.h>
#include "../API/TensorShape.h"
using namespace Cgml;

TensorShape::TensorShape()
{
	setZero();
}

TensorShape::TensorShape( const TensorShape& that )
{
	_mm_storeu_si128( ( __m128i* )size.data(), that.sizeVec() );
	_mm_storeu_si128( ( __m128i* )stride.data(), that.stridesVec() );
}

void TensorShape::operator=( const TensorShape& that )
{
	_mm_storeu_si128( ( __m128i* )size.data(), that.sizeVec() );
	_mm_storeu_si128( ( __m128i* )stride.data(), that.stridesVec() );
}