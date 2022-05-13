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

    float2 pixelOffset = g_View.pixelOffset;
    float samplingRate = ((b_FrameIndex.currentAAMode == nativeResolution || b_FrameIndex.currentAAMode == nativeWithTAA) ? 1.0f : g_SamplingRate.samplingRate);

    const int blockSize = 5;
    const int patchSize = 3;
    const float normalizationFactorPatch = 1.0f / (patchSize * patchSize);
    const float normalizationFactorBlock = 1.0f / (blockSize * blockSize);
    float3 curr[blockSize][blockSize];
    float3 hist[blockSize][blockSize];
    float3 varSqrd[blockSize][blockSize];
    float maximalConfidence[blockSize][blockSize];
    [unroll]
    for (int di = -(blockSize / 2); di <= (blockSize / 2); ++di)
    {
        for (int dj = -(blockSize / 2); dj <= (blockSize / 2); ++dj)
        {
            float3 upsampledJitter = float3(0.0f, 0.0f, 0.0f);
            float2 shiftedIPosition = i_position.xy + float2(di, dj);
            float2 jitterSpaceSVPosition = samplingRate * shiftedIPosition;
            int2 floorSampleIndex = int2(floor(jitterSpaceSVPosition));

            float3 motion_1stmoment = float3(0.0f, 0.0f, 0.0f);
            float maximumWeight = 0.0f;
            float normalizationFactor = 0.0f;

            float3 localPatch1stMoment = float3(0.0f, 0.0f, 0.0f);
            float3 localPatch2ndMoment = float3(0.0f, 0.0f, 0.0f);

            for (int dy = -(patchSize / 2); dy <= (patchSize / 2); ++dy)
            {
                for (int dx = -(patchSize / 2); dx <= (patchSize / 2); ++dx)
                {
                    int2 probedSampleIndex = floorSampleIndex + int2(dx, dy);
                    float2 probedSamplePosition = float2(probedSampleIndex) + float2(0.5f, 0.5f) - pixelOffset;
                    float3 probedJitteredSample = t_JitteredCurrentBuffer[probedSampleIndex].xyz;

                    float probedSampleWeight = tentValue(jitterSpaceSVPosition, probedSamplePosition, samplingRate * 2.0f);

                    localPatch1stMoment += probedJitteredSample;
                    localPatch2ndMoment += probedJitteredSample * probedJitteredSample;

                    upsampledJitter += probedSampleWeight * probedJitteredSample.xyz;
                    normalizationFactor += probedSampleWeight;
                    maximumWeight = max(maximumWeight, probedSampleWeight);

                    float3 proximity_motion = t_MotionVector.Sample(s_LinearSampler, (shiftedIPosition + float2(dx, dy)) * g_View.viewportSizeInv).xyz;

                    float probedSampleLuminance = getLuminance(probedJitteredSample);
                    motion_1stmoment += proximity_motion;
                }
            }
            if (maximumWeight != 0.0f)
            {
                normalizationFactor = 1.0f / normalizationFactor;
                upsampledJitter *= normalizationFactor;
            }
            float3 curr_sample = float3(0.0f, 0.0f, 0.0f);

            int shiftedIndexI = di + blockSize / 2;
            int shiftedIndexJ = dj + blockSize / 2;

            float3 estimatedExpectancy = normalizationFactorPatch * localPatch1stMoment;
            varSqrd[shiftedIndexI][shiftedIndexJ] = normalizationFactorPatch * localPatch2ndMoment - estimatedExpectancy * estimatedExpectancy;

            if (b_FrameIndex.currentAAMode != temporalSupersamplingAA)
            {
                curr_sample = t_JitteredCurrentBuffer[int2(floor(shiftedIPosition * samplingRate))].xyz;
                maximalConfidence[shiftedIndexI][shiftedIndexJ] = 1.0f;
            }
            else
            {
                curr_sample = upsampledJitter.xyz;
                maximalConfidence[shiftedIndexI][shiftedIndexJ] = maximumWeight;
            }
            
            curr[shiftedIndexI][shiftedIndexJ] = curr_sample;

            motion_1stmoment *= normalizationFactorPatch;
            float2 prev_location = shiftedIPosition * g_View.viewportSizeInv - motion_1stmoment.xy;
            //float3 prev_normal = normalize(t_NormalBuffer.Sample(s_LinearSampler, prev_location));
            float3 prev_sample = t_HistoryColor.Sample(s_AnisotropicSampler, prev_location).xyz;
            hist[shiftedIndexI][shiftedIndexJ] = prev_sample;
        }
    }

    float dotProduct = 0.0f;
    float l2NormSqrdCurr = 0.0f;
    float l2NormSqrdHist = 0.0f;
    float maNormSqrdDiff = 0.0f;
    float maximalmaNormSqrd = 0.0f;
    float l1Difference = 0.0f;
    float maximalL1Diff = 0.0f;
    float l2DifferenceSqrd = 0.0f;
    float infDifference = 0.0f;
    float maximalInfDifference = 0.0f;

    const float epsilon = 0.00001f;

    [unroll]
    for (int dk = -(blockSize / 2); dk <= (blockSize / 2); ++dk)
    {
        for (int dl = -(blockSize / 2); dl <= (blockSize / 2); ++dl)
        {
            int shiftedIndexK = dk + blockSize / 2;
            int shiftedIndexL = dl + blockSize / 2;

            float3 currVector = curr[shiftedIndexK][shiftedIndexL];
            float3 histVector = hist[shiftedIndexK][shiftedIndexL];

            dotProduct += dot(currVector, histVector);
            l2NormSqrdCurr += dot(currVector, currVector);
            l2NormSqrdHist += dot(histVector, histVector);

            float3 diffVector = abs(currVector - histVector);
            float3 allOneVector = float3(1.0f, 1.0f, 1.0f);

            l1Difference += dot(diffVector, allOneVector);
            maximalL1Diff += dot(max(currVector, histVector), allOneVector);
            l2DifferenceSqrd += dot(diffVector, diffVector);

            float3 allInvSigmaVector = varSqrd[shiftedIndexK][shiftedIndexL];
            for (int comp = 0; comp < sizeof(allInvSigmaVector) / sizeof(allInvSigmaVector[0]); ++comp)
            {
                float sigmaComponent = 1.0f / allInvSigmaVector[comp];
                allInvSigmaVector[comp] = (sigmaComponent == 0.0f) ? 1.0f : 1.0f / sigmaComponent;
            }
            maNormSqrdDiff += dot(diffVector * diffVector, allInvSigmaVector);
            maximalmaNormSqrd += dot(currVector * currVector, allInvSigmaVector);
            maximalmaNormSqrd += dot(histVector * histVector, allInvSigmaVector);
        }
    }
    float maximalL2DiffSqrd = l2NormSqrdCurr + l2NormSqrdHist;

    //float confidenceFactor = (dotProduct * dotProduct) / (l2NormCurr * l2NormHist + epsilon);//based on dot product
    //float confidenceFactor = 1.0f - l1Difference / (maximalL1Diff + epsilon);//based on l1 correspondence
    float confidenceFactor = 1.0f - l2DifferenceSqrd / (maximalL2DiffSqrd + epsilon);//based on l2 correspondence
    //float confidenceFactor = 1.0f - maNormSqrdDiff / (maximalmaNormSqrd + epsilon);//based on ma correspondence
    //float confidenceFactor = 1.0f;//lmao

    float currentContribution = 0.1f;
    switch (b_FrameIndex.currentAAMode)
    {
    case nativeResolution:
        currentContribution = 1.0f;
        break;
    case rawUpscaled:
        currentContribution = 1.0f;
        break;
    case temporalSupersamplingAA:
        currentContribution *= maximalConfidence[blockSize / 2][blockSize / 2];
        break;
    case temporalAntiAliasingAA:
        currentContribution *= maximalConfidence[blockSize / 2][blockSize / 2];
        break;
    case nativeWithTAA:
        currentContribution *= maximalConfidence[blockSize / 2][blockSize / 2];
        break;
    }

    float3 center_hist = hist[blockSize / 2][blockSize / 2];
    float3 center_curr = curr[blockSize / 2][blockSize / 2];

    //confidenceFactor = 1.0f;
    float historyContribution = (1.0f - currentContribution) * confidenceFactor;
    currentContribution = 1.0f - historyContribution;
    float3 blended = float3(0.0f, 0.0f, 0.0f);
    if (b_FrameIndex.frameHasReset == 0)
    {
        blended = center_hist * historyContribution + center_curr * currentContribution;
    }
    else
    {
        blended = center_curr;
    }

    //blended = varSqrd[blockSize / 2][blockSize / 2];
    //blended = center_curr;
    current_buffer = float4(blended, 1.0f);
    color_buffer = float4(blended, 1.0f);
}
