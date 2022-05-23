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

float cubicBSpline(float2 center, float2 position, float h)
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

    const int blockSize = 5;
    const int patchSize = 5;
    const float normalizationFactorPatch = 1.0f / (patchSize * patchSize);
    const float normalizationFactorBlock = 1.0f / (blockSize * blockSize);
    float3 curr[blockSize][blockSize];
    float3 hist[blockSize][blockSize];
    float3 varSqrdHist[blockSize][blockSize];
    float3 varSqrdLocal[blockSize][blockSize];
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

            float3 localPatch1stMoment = float3(0.0f, 0.0f, 0.0f);
            float3 localPatch2ndMoment = float3(0.0f, 0.0f, 0.0f);

            for (int dy = -(patchSize / 2); dy <= (patchSize / 2); ++dy)
            {
                for (int dx = -(patchSize / 2); dx <= (patchSize / 2); ++dx)
                {
                    int2 probedSampleIndex = floorSampleIndex + int2(dx, dy);
                    float2 probedSamplePosition = float2(probedSampleIndex) + float2(0.5f, 0.5f) - pixelOffset;
                    float3 probedJitteredSample = t_JitteredCurrentBuffer[probedSampleIndex].xyz;

                    //float probedSampleWeight = tentValue(jitterSpaceSVPosition, probedSamplePosition, samplingRate * 1.5f);
                    float probedSampleWeight = cubicBSpline(jitterSpaceSVPosition, probedSamplePosition, samplingRate);

                    localPatch1stMoment += probedJitteredSample;
                    localPatch2ndMoment += probedJitteredSample * probedJitteredSample;

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
                upsampledJitter = t_HistoryColor.Sample(s_LinearSampler, shiftedIPosition * g_View.viewportSizeInv).xyz;
            }
            
            float3 currSample = float3(0.0f, 0.0f, 0.0f);

            int shiftedIndexI = di + blockSize / 2;
            int shiftedIndexJ = dj + blockSize / 2;

            float3 estimatedExpectancy = normalizationFactorPatch * localPatch1stMoment;
            varSqrdLocal[shiftedIndexI][shiftedIndexJ] = normalizationFactorPatch * localPatch2ndMoment - estimatedExpectancy * estimatedExpectancy;

            if (b_FrameIndex.currentAAMode != temporalSupersamplingAA)
            {
                currSample = t_JitteredCurrentBuffer[int2(floor(shiftedIPosition * samplingRate))].xyz;
                maximalConfidence[shiftedIndexI][shiftedIndexJ] = 1.0f;
            }
            else
            {
                currSample = upsampledJitter.xyz;
                maximalConfidence[shiftedIndexI][shiftedIndexJ] = maximumWeight;
                summedConfidence[shiftedIndexI][shiftedIndexJ] = normalizationFactor;;
            }
            
            curr[shiftedIndexI][shiftedIndexJ] = currSample;

            motionFirstMoment *= normalizationFactorPatch;
            prevLocation[shiftedIndexI][shiftedIndexJ] = shiftedIPosition * g_View.viewportSizeInv - motionFirstMoment.xy;
            //float3 prevNormal = normalize(t_NormalBuffer.Sample(s_LinearSampler, prevLocation));
            float3 prevSample = t_HistoryColor.Sample(s_AnisotropicSampler, prevLocation[shiftedIndexI][shiftedIndexJ]).xyz;
            hist[shiftedIndexI][shiftedIndexJ] = prevSample;
            float3 prevExpectancy = t_1stOrderMoment[int2(g_View.viewportSize * prevLocation[shiftedIndexI][shiftedIndexJ])].xyz;
            varSqrdHist[shiftedIndexI][shiftedIndexJ] = t_2ndOrderMoment[int2(g_View.viewportSize * prevLocation[shiftedIndexI][shiftedIndexJ])].xyz - prevExpectancy * prevExpectancy;
        }
    }

    float dotProduct = 0.0f;
    float l2NormSqrdCurr = 0.0f;
    float l2NormSqrdHist = 0.0f;
    float maNormSqrdDiff = 0.0f;
    float tempMaNormSqrdDiff = 0.0f;
    float maximalmaNormSqrd = 0.0f;
    float tempMaximalMaSqrd = 0.0f;
    float l1Difference = 0.0f;
    float maximalL1Diff = 0.0f;
    float l2DifferenceSqrd = 0.0f;
    float infDifference = 0.0f;
    
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

            //float3 diffVector = float3(0.0f, 0.0f, 0.0f);
            float3 diffVector = abs(currVector - histVector);
            /*for (int dm = -1; dm <= 1; ++dm)
            {
                float factorY = (dm == 0) ? 0.5f : 0.25f;
                int convIndexY = dm + dk;
                convIndexY = convIndexY >= 0 ? convIndexY : -convIndexY;
                convIndexY = convIndexY < blockSize ? convIndexY : 2 * blockSize - convIndexY;
                for (int dn = -1; dn <= 1; ++dn)
                {
                    float factorX = (dn == 0) ? 0.5f : 0.25f;
                    int convIndexX = dn + dl;
                    convIndexX = convIndexX >= 0 ? convIndexX : -convIndexX;
                    convIndexX = convIndexX < blockSize ? convIndexX : 2 * blockSize - convIndexX;
                    diffVector += factorX * factorY * abs(hist[convIndexX][convIndexY] - curr[convIndexX][convIndexY]);
                }
            }*/
            float3 allOneVector = float3(1.0f, 1.0f, 1.0f);

            l1Difference += dot(diffVector, allOneVector);
            maximalL1Diff += dot(max(currVector, histVector), allOneVector);
            l2DifferenceSqrd += dot(diffVector, diffVector);
            
            float3 allInvSigmaSpatial = varSqrdLocal[shiftedIndexK][shiftedIndexL];
            float3 allInvSigmaTemporal = varSqrdHist[shiftedIndexK][shiftedIndexL];
            for (int comp = 0; comp < sizeof(allInvSigmaSpatial) / sizeof(allInvSigmaSpatial[0]); ++comp)
            {
                float sigmaComponent = allInvSigmaSpatial[comp];
                allInvSigmaSpatial[comp] = (sigmaComponent == 0.0f) ? 1.0f : 1.0f / sigmaComponent;
                float absDiffComponent = diffVector[comp];
                infDifference = absDiffComponent > infDifference ? absDiffComponent : infDifference;
            }
            for (int compT = 0; compT < sizeof(allInvSigmaTemporal) / sizeof(allInvSigmaTemporal[0]); ++compT)
            {
                float sigmaComponent = allInvSigmaTemporal[compT];
                allInvSigmaTemporal[compT] = (sigmaComponent == 0.0f) ? 1.0f : 1.0f / sigmaComponent;
            }
            maNormSqrdDiff += dot(diffVector * diffVector, allInvSigmaSpatial);
            maximalmaNormSqrd += dot(currVector * currVector, allInvSigmaSpatial);
            maximalmaNormSqrd += dot(histVector * histVector, allInvSigmaSpatial);

            tempMaNormSqrdDiff += dot(diffVector * diffVector, allInvSigmaTemporal);
            tempMaximalMaSqrd += dot(currVector * currVector, allInvSigmaTemporal);
            tempMaximalMaSqrd += dot(histVector * histVector, allInvSigmaTemporal);
        }
    }
    float maximalL2DiffSqrd = l2NormSqrdCurr + l2NormSqrdHist;
    float maximalInfDifference = 1.0f;

    //float confidenceFactor = (dotProduct * dotProduct) / (l2NormCurr * l2NormHist + epsilon);//based on dot product
    //float confidenceFactor = 1.0f - l1Difference / (maximalL1Diff + epsilon);//based on l1 correspondence
    //float confidenceFactor = 1.0f - l2DifferenceSqrd / (maximalL2DiffSqrd + epsilon);//based on l2 correspondence
    //float confidenceFactor = 1.0f - infDifference / maximalInfDifference;//based on linf correspondence
    float confidenceFactor = 1.0f - maNormSqrdDiff / (maximalmaNormSqrd + epsilon);//based on ma correspondence
    //float confidenceFactor = 1.0f - tempMaNormSqrdDiff / (tempMaNormSqrdDiff + epsilon);//based on temporal ma variance
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

    float3 centerHist = hist[blockSize / 2][blockSize / 2];
    float3 centerCurr = curr[blockSize / 2][blockSize / 2];

    float historyContribution = (1.0f - currentContribution) * confidenceFactor;
    currentContribution = 1.0f - historyContribution;
    float3 blended = float3(0.0f, 0.0f, 0.0f);
    if (b_FrameIndex.frameHasReset == 0)
    {
        if (maximalConfidence[blockSize / 2][blockSize / 2] == 0.0f)
        {
            currentContribution = 0.0f;
            historyContribution = 1.0f;
        }
        blended = currentContribution * centerCurr + historyContribution * centerHist;
    }
    
    float3 firstOrderHist = t_1stOrderMoment[int2(g_View.viewportSize * prevLocation[blockSize / 2][blockSize / 2])].xyz;
    float3 secondOrderHist = t_2ndOrderMoment[int2(g_View.viewportSize * prevLocation[blockSize / 2][blockSize / 2])].xyz;
    float sequenceSqrdSumHist = t_SequenceSqrdSum[int2(g_View.viewportSize * prevLocation[blockSize / 2][blockSize / 2])];
    float3 varianceHist = secondOrderHist - firstOrderHist * firstOrderHist;

    float3 firstOrderUpdated = (1.0f - currentContribution) * firstOrderHist + currentContribution * centerCurr;
    float3 secondOrderUpdated = (1.0f - currentContribution) * secondOrderHist + currentContribution * centerCurr * centerCurr;
    
    int2 momentTexelIndex = int2(floor(i_Position.xy));
    float sequenceSqrdSum = sequenceSqrdSumHist * (1.0f - currentContribution) * (1.0f - currentContribution) + currentContribution * currentContribution;
    t_SequenceSqrdSum[momentTexelIndex] = sequenceSqrdSum;
    t_1stOrderMoment[momentTexelIndex] = float4(firstOrderUpdated, 0.0f);
    t_2ndOrderMoment[momentTexelIndex] = float4(secondOrderUpdated, 0.0f);

    float3 expectancy = firstOrderUpdated;
    float3 variance = secondOrderUpdated - expectancy * expectancy;
    float3 fixedLocationDelta = curr[blockSize / 2][blockSize / 2] - expectancy;
    float3 currHistDelta = curr[blockSize / 2][blockSize / 2] - hist[blockSize / 2][blockSize / 2];
    float3 effectiveSamples = 1.0f / sequenceSqrdSum;

    float3 sigmaTolerance = 3.0f;

    o_CurrentBuffer = float4(blended, 1.0f);
    //o_ColorBuffer = float4(confidenceFactor, confidenceFactor, confidenceFactor, 1.0f
    o_ColorBuffer = float4(blended, 1.0f);
}
