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

float catmullRomValue()
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

    const int blockSize = 3;
    const int patchSize = 3;
    const float normalizationFactorPatch = 1.0f / (patchSize * patchSize);
    const float normalizationFactorBlock = 1.0f / (blockSize * blockSize);
    float3 curr[blockSize][blockSize];
    float3 hist[blockSize][blockSize];
    float3 deri[blockSize][blockSize];
    float3 varSqrdSpac[blockSize][blockSize];
    float3 varSqrdTemp[blockSize][blockSize];
    float3 perPixelVelocity[blockSize][blockSize];
    float2 prevLocation[blockSize][blockSize];
    float maximalConfidence[blockSize][blockSize];
    float summedConfidence[blockSize][blockSize];

    [unroll]
    for (int di = -(blockSize / 2); di <= (blockSize / 2); ++di)
    {
        for (int dj = -(blockSize / 2); dj <= (blockSize / 2); ++dj)
        {
            float3 upsampledJitter = float3(0.0f, 0.0f, 0.0f);
            float2 shiftedIPosition = i_Position.xy + float2(di, dj);
            float2 jitterSpaceSVPosition = samplingRate * shiftedIPosition;
            int2 floorSampleIndex = int2(floor(jitterSpaceSVPosition));

            float3 motionFirstMoment = float3(0.0f, 0.0f, 0.0f);
            float maximumWeight = 0.0f;
            float normalizationFactor = 0.0f;

            float3 spatialSecondMoment = 0.0f;
            float3 spatialFirstMoment = 0.0f;

            for (int dy = -(patchSize / 2); dy <= (patchSize / 2); ++dy)
            {
                for (int dx = -(patchSize / 2); dx <= (patchSize / 2); ++dx)
                {
                    int2 probedSampleIndex = floorSampleIndex + int2(dx, dy);
                    float2 probedSamplePosition = float2(probedSampleIndex) + float2(0.5f, 0.5f) - pixelOffset;
                    float3 probedJitteredSample = t_JitteredCurrentBuffer[probedSampleIndex].xyz;

                    float probedSampleWeight = tentValue(jitterSpaceSVPosition, probedSamplePosition, samplingRate * (1.0f + i_Position.z));
                    //float probedSampleWeight = cubicBSplineValue(jitterSpaceSVPosition, probedSamplePosition, samplingRate);

                    upsampledJitter += probedSampleWeight * probedJitteredSample.xyz;
                    normalizationFactor += probedSampleWeight;
                    maximumWeight = max(maximumWeight, probedSampleWeight);

                    float3 proximityMotion = t_MotionVector.Sample(s_LinearSampler, (shiftedIPosition + float2(dx, dy)) * g_View.viewportSizeInv).xyz;

                    float probedSampleLuminance = getLuminance(probedJitteredSample);
                    motionFirstMoment += proximityMotion;
                }
            }
            if (maximumWeight != 0.0f)
            {
                upsampledJitter /= normalizationFactor;
            }
            else
            {
                upsampledJitter = t_JitteredCurrentBuffer.Sample(s_LinearSampler, shiftedIPosition * g_View.viewportSizeInv).xyz;
            }
            
            float3 currSample = float3(0.0f, 0.0f, 0.0f);

            int shiftedIndexI = di + blockSize / 2;
            int shiftedIndexJ = dj + blockSize / 2;

            summedConfidence[shiftedIndexI][shiftedIndexJ] = normalizationFactor;

            if (b_FrameIndex.currentAAMode != temporalSupersamplingAA)
            {
                currSample = t_JitteredCurrentBuffer[int2(floor(shiftedIPosition * samplingRate))].xyz;
                maximalConfidence[shiftedIndexI][shiftedIndexJ] = 1.0f;
            }
            else
            {
                currSample = upsampledJitter.xyz;
                maximalConfidence[shiftedIndexI][shiftedIndexJ] = maximumWeight;
            }
            
            curr[shiftedIndexI][shiftedIndexJ] = currSample;

            //motionFirstMoment = t_MotionVector.Sample(s_LinearSampler, (shiftedIPosition)*g_View.viewportSizeInv).xyz;
            motionFirstMoment *= normalizationFactorPatch;
            perPixelVelocity[shiftedIndexI][shiftedIndexJ] = motionFirstMoment;
            prevLocation[shiftedIndexI][shiftedIndexJ] = shiftedIPosition * g_View.viewportSizeInv - motionFirstMoment.xy;
            float3 prevSample = t_HistoryColor.Sample(s_AnisotropicSampler, prevLocation[shiftedIndexI][shiftedIndexJ]).xyz;
            deri[shiftedIndexI][shiftedIndexJ] = t_HistoryColor.Sample(s_AnisotropicSampler, shiftedIPosition * g_View.viewportSizeInv).xyz;
            if (!isWithInNDC(prevLocation[shiftedIndexI][shiftedIndexJ]))
            {
                prevSample = float3(0.0f, 0.0f, 0.0f);
                maximalConfidence[shiftedIndexI][shiftedIndexJ] = 1.0f;
            }
            hist[shiftedIndexI][shiftedIndexJ] = prevSample;
            float3 prevExpectancy = t_1stOrderMoment[int2(g_View.viewportSize * prevLocation[shiftedIndexI][shiftedIndexJ])].xyz;
            varSqrdTemp[shiftedIndexI][shiftedIndexJ] = t_2ndOrderMoment[int2(g_View.viewportSize * prevLocation[shiftedIndexI][shiftedIndexJ])].xyz - prevExpectancy * prevExpectancy;

            for (int dyt = -(patchSize / 2); dyt <= (patchSize / 2); ++dyt)
            {
                for (int dxt = -(patchSize / 2); dxt <= (patchSize / 2); ++dxt)
                {
                    float3 probedHistSample = t_HistoryColor.Sample(s_AnisotropicSampler, prevLocation[shiftedIndexI][shiftedIndexJ] + g_View.viewportSizeInv * float2(dxt, dyt)).xyz;

                    spatialSecondMoment += probedHistSample * probedHistSample;
                    spatialFirstMoment += probedHistSample;
                }
            }
            spatialFirstMoment *= normalizationFactorPatch;
            varSqrdSpac[shiftedIndexI][shiftedIndexJ] = spatialSecondMoment * normalizationFactorPatch - spatialFirstMoment * spatialFirstMoment;
        }
    }

    float tempMaNormSqrdDiff = 0.0f;
    float tempMaximalMaSqrd = 0.0f;
    
    const float epsilon = 0.00001f;

    float3 minimalCurr = float3(1.0f, 1.0f, 1.0f);
    float3 maximalCurr = float3(0.0f, 0.0f, 0.0f);
    [unroll]
    for (int dk = -(blockSize / 2); dk <= (blockSize / 2); ++dk)
    {
        for (int dl = -(blockSize / 2); dl <= (blockSize / 2); ++dl)
        {
            int shiftedIndexK = dk + blockSize / 2;
            int shiftedIndexL = dl + blockSize / 2;

            if (maximalConfidence[shiftedIndexK][shiftedIndexL] == 0.0f)
            {
                continue;
            }
            
            float3 currVector = curr[shiftedIndexK][shiftedIndexL];
            float3 histVector = hist[shiftedIndexK][shiftedIndexL];
            float3 deriVector = deri[shiftedIndexK][shiftedIndexL];
            float3 diffVector = abs(currVector - histVector) * maximalConfidence[shiftedIndexK][shiftedIndexL] + (1.0f - maximalConfidence[shiftedIndexK][shiftedIndexL]) * abs(deriVector - histVector);
            
            minimalCurr = min(minimalCurr, currVector);
            maximalCurr = max(maximalCurr, currVector);
            //float3 allInvSigmaTemporal = float3(1.0f, 1.0f, 1.0f);
            float3 allInvSigmaTemporal = varSqrdTemp[shiftedIndexK][shiftedIndexL];
            for (int compT = 0; compT < sizeof(allInvSigmaTemporal) / sizeof(allInvSigmaTemporal[0]); ++compT)
            {
                float sigmaComponent = allInvSigmaTemporal[compT];
                allInvSigmaTemporal[compT] = (sigmaComponent < epsilon) ? 1.0f / epsilon : 1.0f / sigmaComponent;
            }

            tempMaNormSqrdDiff += dot(diffVector * diffVector, allInvSigmaTemporal);
            tempMaximalMaSqrd += dot(currVector * currVector, allInvSigmaTemporal);
            tempMaximalMaSqrd += dot(histVector * histVector, allInvSigmaTemporal);
        }
    }

    float confidenceFactor = 1.0f - sqrt(tempMaNormSqrdDiff) / (sqrt(tempMaximalMaSqrd) + epsilon);//based on ma correspondence
    //confidenceFactor *= 1.0f - length(perPixelVelocity[blockSize / 2][blockSize / 2]);
    //confidenceFactor = log(confidenceFactor);
    //confidenceFactor *= confidenceFactor;
    //confidenceFactor *= confidenceFactor * confidenceFactor;
    float2 prevLocationCriterion = prevLocation[blockSize / 2][blockSize / 2];
    if (!isWithInNDC(prevLocation[blockSize / 2][blockSize / 2]))
    {
        confidenceFactor = 0.0f;
    }
    //confidenceFactor = 1.0f;

    int2 momentTexelIndex = int2(floor(i_Position.xy));
    float summedAlphaSqrd = t_SequenceSqrdSum[momentTexelIndex];
    summedAlphaSqrd = summedAlphaSqrd < epsilon ? 1.0f : summedAlphaSqrd;
    float effectiveSamples = 1.0f / summedAlphaSqrd;
    //float currentContribution = 0.1f;
    float currentContribution = 1.0f / (effectiveSamples + 1.0f);
    switch (b_FrameIndex.currentAAMode)
    {
    case nativeResolution:
        currentContribution = 1.0f;
        break;
    case rawUpscaled:
        currentContribution = 1.0f;
        break;
    case temporalSupersamplingAA:
        //currentContribution *= maximalConfidence[blockSize / 2][blockSize / 2];
        break;
    case temporalAntiAliasingAA:
        //currentContribution *= maximalConfidence[blockSize / 2][blockSize / 2];
        break;
    case nativeWithTAA:
        //currentContribution *= maximalConfidence[blockSize / 2][blockSize / 2];
        break;
    }

    float3 centerHist = hist[blockSize / 2][blockSize / 2];
    float3 centerCurr = curr[blockSize / 2][blockSize / 2];
    //centerHist = max(min(centerHist, maximalCurr), minimalCurr);

    float historyContribution = (1.0f - currentContribution) * confidenceFactor;
    currentContribution = 1.0f - historyContribution;
    float3 blended = float3(0.0f, 0.0f, 0.0f);
    if (b_FrameIndex.frameHasReset == 0)
    {
        if (maximalConfidence[blockSize / 2][blockSize / 2] < epsilon)
        {
            currentContribution = 0.0f;
            historyContribution = 1.0f;
        }
        blended = currentContribution * centerCurr + historyContribution * centerHist;
    }
    /*if (isWithInNDC(prevLocation[blockSize / 2][blockSize / 2]))
    {
        blended += maximalConfidence[blockSize / 2][blockSize / 2] * (deri[blockSize / 2][blockSize / 2] - hist[blockSize / 2][blockSize / 2]);
    }*/
    
    float3 firstOrderHist = t_1stOrderMoment[int2(g_View.viewportSize * prevLocation[blockSize / 2][blockSize / 2])].xyz;
    float3 secondOrderHist = t_2ndOrderMoment[int2(g_View.viewportSize * prevLocation[blockSize / 2][blockSize / 2])].xyz;
    float sequenceSqrdSumHist = t_SequenceSqrdSum[int2(g_View.viewportSize * prevLocation[blockSize / 2][blockSize / 2])];
    float3 varianceHist = secondOrderHist - firstOrderHist * firstOrderHist;

    float3 firstOrderUpdated = (1.0f - currentContribution) * firstOrderHist + currentContribution * centerCurr;
    float3 secondOrderUpdated = (1.0f - currentContribution) * secondOrderHist + currentContribution * centerCurr * centerCurr;
    
    float sequenceSqrdSum = sequenceSqrdSumHist * (1.0f - currentContribution) * (1.0f - currentContribution) + currentContribution * currentContribution;
    t_SequenceSqrdSum[momentTexelIndex] = sequenceSqrdSum;
    t_1stOrderMoment[momentTexelIndex] = float4(firstOrderUpdated, 0.0f);
    t_2ndOrderMoment[momentTexelIndex] = float4(secondOrderUpdated, 0.0f);

    float3 expectancy = firstOrderUpdated;
    float3 variance = secondOrderUpdated - expectancy * expectancy;
    float3 fixedLocationDelta = curr[blockSize / 2][blockSize / 2] - expectancy;
    float3 currHistDelta = curr[blockSize / 2][blockSize / 2] - hist[blockSize / 2][blockSize / 2];
    
    o_CurrentBuffer = float4(blended, 1.0f);
    o_ColorBuffer = float4(blended, 1.0f);
}
