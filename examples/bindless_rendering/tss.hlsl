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
Texture2D<float> t_Coverage : register(t5);

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

float gaussValue(int dx, int dy)
{
    float dxContribution = (dx == 0 ? 0.5f : 0.25f);
    float dyContribution = (dy == 0 ? 0.5f : 0.25f);
    return dxContribution * dyContribution;
}

float tentValue(float2 center, float2 position)
{
    float2 diff = abs(position - center);
    float contributionX = (diff.x > 0.25f ? 0.0f : 1.0f - 4.0f * diff.x);
    float contributionY = (diff.y > 0.25f ? 0.0f : 1.0f - 4.0f * diff.y);
    return contributionX * contributionY;
}

float4 tentSampling(float2 svPosition, Texture2D<float4> sourceTexture)
{
    float samplingRate = g_SamplingRate.samplingRate;
    float2 pixelOffset = g_View.pixelOffset;

    float2 pixelPosition = svPosition;
    float2 pixelPositionJitterSpace = pixelPosition * samplingRate;
    int2 closestJitterCellIndex = int2(floor(pixelPositionJitterSpace.x), floor(pixelPositionJitterSpace.y));
    float2 closestJitterSamplePosition = float2(closestJitterCellIndex.x, closestJitterCellIndex.y) + float2(0.5f, 0.5f) + pixelOffset;

    float normalizationFactor = 0.0f;
    float maximalWeight = 0.0f;
    float3 collectedSample = float3(0.0f, 0.0f, 0.0f);
    const int patch_size = 3;
    [unroll]
    for (int dy = -(patch_size / 2); dy <= (patch_size / 2); ++dy)
    {
        for (int dx = -(patch_size / 2); dx <= (patch_size / 2); ++dx)
        {
            float2 jitterSamplePosition = float2(dx, dy) + closestJitterSamplePosition;
            //float sampleWeight = tentValue(pixelPositionJitterSpace, jitterSamplePosition);
            float sampleWeight = gaussValue(dx, dy);
            normalizationFactor += sampleWeight;
            maximalWeight = max(maximalWeight, sampleWeight);

            float2 jitterTextureCoordinate = jitterSamplePosition * g_View.viewportSizeInv * (1.0f / samplingRate);
            float3 jitterSample = sourceTexture.Sample(s_FrameSampler, jitterTextureCoordinate).xyz;
            collectedSample += sampleWeight * jitterSample;
        }
    }
    
    if (maximalWeight != 0.0f)
    {
        normalizationFactor = 1.0f / normalizationFactor;
        return float4(collectedSample.xyz * normalizationFactor, maximalWeight);
    }
    else
    {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

void ps_main(
    in float4 i_position : SV_Position,
    in float2 i_uv_coord : TEXTURE_COORD,
    out float4 color_buffer : SV_Target0,
    out float4 current_buffer : SV_Target1)
{
    float maximalConfidence = 1.0f;
    float3 curr = float3(0.0f, 0.0f, 0.0f);
    if (b_FrameIndex.currentAAMode != 2)
    {
        curr = t_JitteredCurrentBuffer.Sample(s_FrameSampler, i_position.xy * g_View.viewportSizeInv).xyz;
    }
    else
    {
        float4 tent_result = tentSampling(i_position.xy, t_JitteredCurrentBuffer);
        curr = tent_result.xyz;
        maximalConfidence = tent_result.w;
    }

    float3 motion_1stmoment = float3(0.0f, 0.0f, 0.0f);
    const int patch_size = 3;
    [unroll]
    for (int dy = -(patch_size / 2); dy <= (patch_size / 2); ++dy)
    {
        for (int dx = -(patch_size / 2); dx <= (patch_size / 2); ++dx)
        {
            float2 probing_index = i_position.xy + float2(dx, dy);
            probing_index = clamp(probing_index, int2(0, 0), g_View.viewportOrigin + g_View.viewportSize);
            float3 proximity_motion = t_MotionVector.Sample(s_FrameSampler, probing_index * g_View.viewportSizeInv).xyz;

            motion_1stmoment += proximity_motion;
        }
    }

    const float normalization_factor = 1.0f / (patch_size * patch_size);
    motion_1stmoment *= normalization_factor;
    
    float3 blended = curr;
    if (b_FrameIndex.frameHasReset == 0 && b_FrameIndex.currentAAMode == 2)
    {
        float2 prev_location = i_position.xy - motion_1stmoment.xy * g_View.viewportSize;
        if (all(prev_location > g_View.viewportOrigin) && all(prev_location < g_View.viewportOrigin + g_View.viewportSize))
        {
            prev_location *= g_View.viewportSizeInv;
            float3 hist = t_HistoryColor.Sample(s_FrameSampler, prev_location).xyz;
            float alphaConfidence = 0.9f * maximalConfidence;
            blended = (1.0f - alphaConfidence) * hist + alphaConfidence * curr;
        }
    }

    float isCovered = t_Coverage[int2(floor(i_position.x), floor(i_position.y))];
    if (isCovered == 0.0f)
    {
        current_buffer = float4(curr, 1.0f);
        color_buffer = float4(curr, 1.0f);
    }
    else
    {
        current_buffer = float4(blended, 1.0f);
        color_buffer = float4(blended, 1.0f);
    }
}
