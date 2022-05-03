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

SamplerState s_AnisotropicSampler : register(s0);
SamplerState s_LinearSampler : register(s1);
SamplerState s_NearestSampler : register(s2);

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

void ps_main(
    in float4 i_position : SV_Position,
    in float2 i_uv_coord : TEXTURE_COORD,
    out float4 color_buffer : SV_Target0,
    out float4 current_buffer : SV_Target1)
{
    const int nativeResolution = 0;
    const int nativeWithTAA = 1;
    const int rawUpscaled = 2;
    const int temporalSupersamplingAA = 3;
    const int temporalAntiAliasingAA = 4;

    float maximalConfidence;
    float3 curr = float3(0.0f, 0.0f, 0.0f);
    //float3 curr_normal = t_NormalBuffer.Sample(s_LinearSampler, i_position.xy * g_View.viewportSizeInv);
    float2 pixelOffset = g_View.pixelOffset;
    float samplingRate = ((b_FrameIndex.currentAAMode == nativeResolution || b_FrameIndex.currentAAMode == nativeWithTAA) ? 1.0f : g_SamplingRate.samplingRate);

    //float3 color_1stmoment = float3(0.0f, 0.0f, 0.0f);
    //float3 color_2ndmoment = float3(0.0f, 0.0f, 0.0f);
    float3 color_lowerbound = float3(1.0f, 1.0f, 1.0f);
    float3 color_upperbound = float3(0.0f, 0.0f, 0.0f);
    float luminanceLowerbound = 1.0f;
    float luminanceUpperbound = 0.0f;

    float3 motion_1stmoment = float3(0.0f, 0.0f, 0.0f);
    
    float3 upsampledJitter = float3(0.0f, 0.0f, 0.0f);
    float2 jitterSpaceSVPosition = samplingRate * i_position.xy;
    int2 floorSampleIndex = int2(floor(jitterSpaceSVPosition.x), floor(jitterSpaceSVPosition.y));
    float maximumWeight = 0.0f;
    float normalizationFactor = 0.0f;

    const int patchSize = 3;
    [unroll]
    for (int dy = -(patchSize / 2); dy <= (patchSize / 2); ++dy)
    {
        for (int dx = -(patchSize / 2); dx <= (patchSize / 2); ++dx)
        {
            int2 probedSampleIndex = floorSampleIndex + int2(dx, dy);
            float2 probedSamplePosition = float2(probedSampleIndex) + float2(0.5f, 0.5f) - pixelOffset;
            float3 probedJitteredSample = t_JitteredCurrentBuffer[probedSampleIndex].xyz;
            float probedSampleWeight = tentValue(jitterSpaceSVPosition, probedSamplePosition, samplingRate * 2.0f);

            upsampledJitter += probedSampleWeight * probedJitteredSample.xyz;
            normalizationFactor += probedSampleWeight;
            maximumWeight = max(maximumWeight, probedSampleWeight);

            float3 proximity_motion = t_MotionVector.Sample(s_LinearSampler, (i_position.xy + float2(dx, dy)) * g_View.viewportSizeInv).xyz;

            //color_1stmoment += probedJitteredSample;
            //color_2ndmoment += probedJitteredSample * probedJitteredSample;
            float probedSampleLuminance = getLuminance(probedJitteredSample);
            color_lowerbound = min(probedJitteredSample, color_lowerbound);
            color_upperbound = max(probedJitteredSample, color_upperbound);
            luminanceLowerbound = min(luminanceLowerbound, probedSampleLuminance);
            luminanceUpperbound = max(luminanceUpperbound, probedSampleLuminance);

            motion_1stmoment += proximity_motion;
        }
    }
    if (maximumWeight == 0.0f)
    {
        upsampledJitter = float3(0.0f, 0.0f, 0.0f);
    }
    else
    {
        normalizationFactor = 1.0f / normalizationFactor;
        upsampledJitter *= normalizationFactor;
    }

    if (b_FrameIndex.currentAAMode != temporalSupersamplingAA)
    {
        curr = t_JitteredCurrentBuffer[int2(floor(i_position.x * samplingRate), floor(i_position.y * samplingRate))].xyz;
        maximalConfidence = 1.0f;
    }
    else
    {
        curr = upsampledJitter.xyz;
        maximalConfidence = maximumWeight;
    }

    const float normalizationFactorPatch = 1.0f / (patchSize * patchSize);
    motion_1stmoment *= normalizationFactorPatch;
    //color_1stmoment *= normalizationFactorPatch;
    //color_2ndmoment *= normalizationFactorPatch;
    //float3 color_var = color_2ndmoment - color_1stmoment * color_1stmoment;
    //const float var_magnitude = 5.0f;
    //float3 color_width = sqrt(color_var) * var_magnitude;
    //color_lowerbound = max(curr - color_width, float3(0.0f, 0.0f, 0.0f));
    //color_upperbound = min(curr + color_width, float3(1.0f, 1.0f, 1.0f));

    float2 prev_location = i_position.xy * g_View.viewportSizeInv - motion_1stmoment.xy;
    float3 prev_normal = normalize(t_NormalBuffer.Sample(s_LinearSampler, prev_location));
    
    float3 blended = float3(0.0f, 0.0f, 0.0f);
    float blendAlpha = 1.0f;
    switch (b_FrameIndex.currentAAMode)
    {
    case nativeResolution:
        blendAlpha = 1.0f;
        break;
    case rawUpscaled:
        blendAlpha = 1.0f;
        break;
    case temporalSupersamplingAA:
        blendAlpha = maximalConfidence * 0.1f;
        break;
    case temporalAntiAliasingAA:
        blendAlpha = maximalConfidence * 0.1f;
        break;
    case nativeWithTAA:
        blendAlpha = maximalConfidence * 0.1f;
        break;
    }
    float3 hist = t_HistoryColor.Sample(s_AnisotropicSampler, prev_location).xyz;
    if (b_FrameIndex.frameHasReset == 0)
    {
        //if (all(prev_location >= float2(0.0f, 0.0f)) && all(prev_location <= float2(1.0f, 1.0f)))
        {
            //float histLuminance = getLuminance(hist);
            //blendAlpha = (histLuminance > luminanceLowerbound && histLuminance < luminanceUpperbound) ? blendAlpha : 1.0f;
            //hist = max(color_lowerbound, hist);
            //hist = min(color_upperbound, hist);

            blended = lerp(hist, curr, blendAlpha);
        }
    }
    //blended = color_upperbound;
    current_buffer = float4(blended, 1.0f);
    color_buffer = float4(blended, 1.0f);
}
