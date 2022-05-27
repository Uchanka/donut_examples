/*
* Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#pragma pack_matrix(row_major)

#include <donut/shaders/bindless.h>
#include <donut/shaders/view_cb.h>

#ifdef SPIRV
#define VK_PUSH_CONSTANT [[vk::push_constant]]
#define VK_BINDING(reg,dset) [[vk::binding(reg,dset)]]
#else
#define VK_PUSH_CONSTANT
#define VK_BINDING(reg,dset) 
#endif

ConstantBuffer<PlanarViewConstants> g_View : register(b0);
ConstantBuffer<SamplingRateWrapper> g_SamplingRate : register(b1);
VK_PUSH_CONSTANT ConstantBuffer<FrameIndexConstant> b_FrameIndex : register(b2);
Texture2D<float4> t_MotionVector : register(t0);
Texture2D<float4> t_HistoryColor : register(t1);
Texture2D<float4> t_JitteredCurrentBuffer : register(t2);
Texture2D<float3> t_NormalBuffer : register(t3);
Texture2D<float3> t_HistoryNormal : register(t4);

RWTexture2D<float> t_SequenceSqrdSum : register(u0);
RWTexture2D<float4> t_1stOrderMoment : register(u1);
RWTexture2D<float4> t_2ndOrderMoment : register(u2);

SamplerState s_AnisotropicSampler : register(s0);
SamplerState s_LinearSampler : register(s1);
SamplerState s_NearestSampler : register(s2);

static const float2 g_positions[] =
{
    float2(1.0f, 3.0f),
    float2(-3.0f, -1.0f),
    float2(1.0f, -1.0f)
};

static const float2 g_uvs[] =
{
    float2(1.0f, 2.0f),
    float2(-1.0f, 0.0f),
    float2(1.0f, 0.0f)
};

float getLuminance(float3 color)
{
    return 0.30f * color.r + 0.59f * color.g + 0.11f * color.b;
}

void vs_main(
    in uint i_vertexID : SV_VertexID,
    out float4 o_position : SV_Position,
    out float2 o_uv_coord : TEXTURE_COORD)
{
    o_position = float4(g_positions[i_vertexID], 0.0f, 1.0f);
    o_uv_coord = g_uvs[i_vertexID];
}

float tentValue(float2 center, float2 position, float tentWidth)
{
    float2 diff = abs(position - center);
    float k = 1.0f / (0.5f * tentWidth);
    float contributionX = (diff.x > 0.5f * tentWidth ? 0.0f : 1.0f - k * diff.x);
    contributionX = clamp(contributionX, 0.0f, 1.0f);
    float contributionY = (diff.y > 0.5f * tentWidth ? 0.0f : 1.0f - k * diff.y);
    contributionY = clamp(contributionY, 0.0f, 1.0f);
    return contributionX * contributionY;
}

float cubicBSplineValue(float2 center, float2 position, float h)
{
    float2 x = (position - center) / h;
    float2 absx = abs(x);
    float2 xsqr = x * x;
    float2 absxcubic = xsqr * absx;
    float Nx = 0.0f;
    float Ny = 0.0f;

    if (absx.x < 1.0f)
    {
        Nx = 0.5 * absxcubic.x - xsqr.x + 2.0f / 3.0f;
    }
    else if (absx.x < 2.0f)
    {
        Nx = -1.0f / 6.0f * absxcubic.x + xsqr.x - 2.0f * absx.x + 4.0f / 3.0f;
    }
    if (absx.y < 1.0f)
    {
        Ny = 0.5 * absxcubic.y - xsqr.y + 2.0f / 3.0f;
    }
    else if (absx.y < 2.0f)
    {
        Ny = -1.0f / 6.0f * absxcubic.y + xsqr.y - 2.0f * absx.y + 4.0f / 3.0f;
    }

    return Nx * Ny;
}

float catmullRomValue(float2 center, float2 position, float h)
{
    return 0.0f;
}

bool isWithInNDC(float2 ndcCoordinates)
{
    if (ndcCoordinates.x < 0.0f || ndcCoordinates.y < 0.0f || ndcCoordinates.x > 1.0f || ndcCoordinates.y > 1.0f)
    {
        return false;
    }
    else
    {
        return true;
    }
}

void ps_main(
    in float4 i_Position : SV_Position,
    out float4 o_ColorBuffer : SV_Target0,
    out float4 o_CurrentBuffer : SV_Target1)
{
    const int nativeResolution = 0;
    const int nativeWithTAA = 1;
    const int rawUpscaled = 2;
    const int temporalSupersamplingAA = 3;
    const int temporalAntiAliasingAA = 4;

    float2 pixelOffset = g_View.pixelOffset;
    float samplingRate = ((b_FrameIndex.currentAAMode == nativeResolution || b_FrameIndex.currentAAMode == nativeWithTAA) ? 1.0f : g_SamplingRate.samplingRate);


    
    o_CurrentBuffer = float4(blended, 1.0f);
    o_ColorBuffer = float4(blended, 1.0f);
}
