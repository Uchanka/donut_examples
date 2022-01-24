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

SamplerState s_FrameSampler : register(s0);

static const float2 g_positions[] =
{
    float2(-1.0f, 1.0f),
    float2(1.0f, 1.0f),
    float2(-1.0f, -1.0f),

    float2(1.0f, 1.0f),
    float2(-1.0f, -1.0f),
    float2(1.0f, -1.0f)
};

static const float2 g_uvs[] =
{
    float2(0.0f, 0.0f),
    float2(1.0f, 0.0f),
    float2(0.0f, 1.0f),

    float2(1.0f, 0.0f),
    float2(0.0f, 1.0f),
    float2(1.0f, 1.0f)
};

void vs_main(
    in uint i_vertexID : SV_VertexID,
    out float4 o_position : SV_Position,
    out float2 o_uv_coord : TEXTURE_COORD)
{
    o_position = float4(g_positions[i_vertexID], 0.0f, 1.0f);
    o_uv_coord = g_uvs[i_vertexID];
}

float tentValue(float2 center, float2 position)
{
    float2 diff = abs(position - center);
    float contributionX = (diff.x == 0.0f ? 1.0f : 0.0f);
    float contributionY = (diff.y == 0.0f ? 1.0f : 0.0f);
    return contributionX * contributionY;
}

float4 tentSampling(float2 svPosition, Texture2D<float4> sourceTexture)
{
    float samplingRate = g_SamplingRate.samplingRate;
    float2 pixelOffset = g_View.pixelOffset;

    return float4(0.0f, 0.0f, 0.0f, 0.0f);
}

void ps_main(
    in float4 i_position : SV_Position,
    in float2 i_uv_coord : TEXTURE_COORD,
    out float4 color_buffer : SV_Target0,
    out float4 current_buffer : SV_Target1)
{
    int2 pixelIndex = int2(floor(i_position.x), floor(i_position.y));
    float samplingRate = (b_FrameIndex.currentAAMode == 2 ? g_SamplingRate.samplingRate : 1.0f);
    int2 jitterIndex = int2(floor(i_position.x * samplingRate), floor(i_position.y * samplingRate));

    float4 motionVector = t_MotionVector.Sample(s_FrameSampler, i_position.xy * g_View.viewportSizeInv) * g_SamplingRate.samplingRate;
    float2 prevLocation = i_position.xy - motionVector.xy;

    //float4 historySample = t_HistoryColor.Sample(s_FrameSampler, prevLocation * g_View.viewportSizeInv);
    float4 historySample = t_HistoryColor[pixelIndex];
    float4 jitteredSample = t_JitteredCurrentBuffer[jitterIndex];

    int evenOddXIndex = (pixelIndex.x % 2 == 0 ? 0 : 1);
    int evenOddYIndex = (pixelIndex.y % 2 == 0 ? 0 : 1);
    int evenOddIndex = evenOddXIndex + (evenOddYIndex << 1);

    float2 jitterOffset = g_View.pixelOffset;
    float4 resultSample = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (b_FrameIndex.currentAAMode == 0)
    {

    }
    else
    {
        switch (evenOddIndex)
        {
        case 0:
            if (jitterOffset.x == 0.25f && jitterOffset.y == 0.25f)
            {
                resultSample = jitteredSample;
            }
            else
            {
                resultSample = historySample;
            }
            break;
        case 1:
            if (jitterOffset.x == -0.25f && jitterOffset.y == 0.25f)
            {
                resultSample = jitteredSample;
            }
            else
            {
                resultSample = historySample;
            }
            break;
        case 2:
            if (jitterOffset.x == 0.25f && jitterOffset.y == -0.25f)
            {
                resultSample = jitteredSample;
            }
            else
            {
                resultSample = historySample;
            }
            break;
        case 3:
            if (jitterOffset.x == -0.25f && jitterOffset.y == -0.25f)
            {
                resultSample = jitteredSample;
            }
            else
            {
                resultSample = historySample;
            }
            break;
        }
    }

    //float alphaCoeff = (b_FrameIndex.currentAAMode != 2 ? 1.0f : 0.3f);
    //float4 blendedSample = lerp(historySample, jitteredSample, alphaCoeff);

    current_buffer = float4(resultSample.xyz, 1.0f);
    color_buffer = float4(resultSample.xyz, 1.0f);
}
