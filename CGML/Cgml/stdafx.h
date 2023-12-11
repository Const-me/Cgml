#pragma once
#include <stdint.h>
#include <assert.h>
#include <immintrin.h>
#include <vector>
#include <array>
#include <string>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <atlbase.h>

#include "Utils/Logger.h"
#include "Utils/miscUtils.h"
// CHECK() macro is there
#include "../ComLightLib/hresult.h"