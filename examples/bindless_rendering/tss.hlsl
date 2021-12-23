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

struct FrameIndexConstant
{
    int frameIndex;
    int taaEnabled;
};

ConstantBuffer<PlanarViewConstants> g_View : register(b0);
VK_PUSH_CONSTANT ConstantBuffer<FrameIndexConstant> b_FrameIndex : register(b1);
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

void ps_main(
    in float4 i_position : SV_Position,
    in float2 i_uv_coord : TEXTURE_COORD,
    out float4 color_buffer : SV_Target0,
    out float4 current_buffer : SV_Target1)
{
    float3 curr = t_JitteredCurrentBuffer.Sample(s_FrameSampler, i_position.xy * g_View.viewportSizeInv).xyz;
    float3 curr_normal = t_NormalBuffer[i_position.xy];

    float3 color_1stmoment = float3(0.0f, 0.0f, 0.0f);
    float3 color_2ndmoment = float3(0.0f, 0.0f, 0.0f);
    float3 color_lowerbound = float3(1.0f, 1.0f, 1.0f);
    float3 color_upperbound = float3(0.0f, 0.0f, 0.0f);

    float3 motion_1stmoment = float3(0.0f, 0.0f, 0.0f);
    const int patch_size = 3;
    [unroll]
    for (int dy = -(patch_size / 2); dy <= (patch_size / 2); ++dy)
    {
        for (int dx = -(patch_size / 2); dx <= (patch_size / 2); ++dx)
        {
            int2 probing_index = i_position.xy + int2(dx, dy);
            probing_index = clamp(probing_index, int2(0, 0), g_View.viewportOrigin + g_View.viewportSize);
            float3 proximity_color = t_JitteredCurrentBuffer[probing_index].xyz;
            float3 proximity_motion = t_MotionVector[probing_index].xyz;

            color_1stmoment += proximity_color;
            color_2ndmoment += proximity_color * proximity_color;
            color_lowerbound = min(proximity_color, color_lowerbound);
            color_upperbound = max(proximity_color, color_upperbound);

            motion_1stmoment += proximity_motion;
        }
    }

    const float normalization_factor = 1.0f / (patch_size * patch_size);
    color_1stmoment *= normalization_factor;
    color_2ndmoment *= normalization_factor;
    motion_1stmoment *= normalization_factor;
    float3 color_var = color_2ndmoment - color_1stmoment * color_1stmoment;
    const float var_magnitude = 5.0f;
    float3 color_width = sqrt(color_var) * var_magnitude;
    //color_lowerbound = max(curr - color_width, float3(0.0f, 0.0f, 0.0f));
    //color_upperbound = min(curr + color_width, float3(1.0f, 1.0f, 1.0f));
    
    float3 blended = curr;
    if (b_FrameIndex.frameIndex != 0 && b_FrameIndex.taaEnabled)
    {
        float2 prev_location = i_position.xy - motion_1stmoment.xy * g_View.viewportSize;
        if (all(prev_location > g_View.viewportOrigin) && all(prev_location < g_View.viewportOrigin + g_View.viewportSize))
        {
            prev_location /= g_View.viewportSize;
            float3 prev_normal = normalize(t_NormalBuffer.Sample(s_FrameSampler, prev_location));
            if (pow(dot(prev_normal, curr_normal), 32) > 0.80f && abs(motion_1stmoment.z) < 0.05f)
            {
                float3 hist = t_HistoryColor.Sample(s_FrameSampler, prev_location).xyz;
                hist = max(color_lowerbound, hist);
                hist = min(color_upperbound, hist);

                blended = lerp(curr, hist, 0.9f);
            }
        }
    }
   
    current_buffer = float4(blended, 1.0f);
    color_buffer = float4(blended, 1.0f);
}
