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
    out float4 current_buffer : SV_Target3)
{
    float3 curr = t_JitteredCurrentBuffer[i_position.xy].xyz;
    
    current_buffer = float4(curr, 1.0f);
    color_buffer = float4(curr, 1.0f);
}
