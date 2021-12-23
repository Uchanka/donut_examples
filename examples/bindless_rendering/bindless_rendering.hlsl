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
#include <donut/shaders/packing.hlsli>

#ifdef SPIRV
#define VK_PUSH_CONSTANT [[vk::push_constant]]
#define VK_BINDING(reg,dset) [[vk::binding(reg,dset)]]
#else
#define VK_PUSH_CONSTANT
#define VK_BINDING(reg,dset) 
#endif

struct InstanceConstants
{
    uint instance;
    uint geometryInMesh;
};

ConstantBuffer<PlanarViewConstants> g_View : register(b0);
ConstantBuffer<PlanarViewConstants> g_ViewLastFrame : register(b1);
VK_PUSH_CONSTANT ConstantBuffer<InstanceConstants> g_Instance : register(b2);

StructuredBuffer<InstanceData> t_InstanceData : register(t0);
StructuredBuffer<GeometryData> t_GeometryData : register(t1);
StructuredBuffer<MaterialConstants> t_MaterialConstants : register(t2);

SamplerState s_MaterialSampler : register(s0);

VK_BINDING(0, 1) ByteAddressBuffer t_BindlessBuffers[] : register(t0, space1);
VK_BINDING(1, 1) Texture2D t_BindlessTextures[] : register(t0, space2);

void vs_main(
    in uint i_vertexID : SV_VertexID,
    out float4 o_position : SV_Position,
    out float4 o_curr_position : CURR_POSITION,
    out float4 o_prev_position : PREV_POSITION,
    out float3 o_normal_vector : NORMAL_VECTOR,
    out float3 o_prev_normal : PREV_NORMAL,
    out float2 o_uv : TEXCOORD,
    out uint o_material : MATERIAL)
{
    InstanceData instance = t_InstanceData[g_Instance.instance];
    GeometryData geometry = t_GeometryData[instance.firstGeometryIndex + g_Instance.geometryInMesh];

    ByteAddressBuffer indexBuffer = t_BindlessBuffers[geometry.indexBufferIndex];
    ByteAddressBuffer vertexBuffer = t_BindlessBuffers[geometry.vertexBufferIndex];

    uint index = indexBuffer.Load(geometry.indexOffset + i_vertexID * 4);

    float2 texcoord = geometry.texCoord1Offset == ~0u ? 0 : asfloat(vertexBuffer.Load2(geometry.texCoord1Offset + index * 8));
    float3 objectSpacePosition = asfloat(vertexBuffer.Load3(geometry.positionOffset + index * 12));
    float4 objectSpaceNormal = Unpack_RGBA8_SNORM(vertexBuffer.Load(geometry.normalOffset + index * 4));
    o_normal_vector = abs(mul(objectSpaceNormal, g_View.matWorldToViewNormal)).xyz;
    o_prev_normal = abs(mul(objectSpaceNormal, g_ViewLastFrame.matWorldToViewNormal)).xyz;

    float3 worldSpacePosition = mul(instance.transform, float4(objectSpacePosition, 1.0f)).xyz;
    float4 clipSpacePosition = mul(float4(worldSpacePosition, 1.0f), g_View.matWorldToClip);
    float4 clipSpacePositionNoOffset = mul(float4(worldSpacePosition, 1.0f), g_View.matWorldToClipNoOffset);

    float3 worldSpacePositionLastFrame = mul(instance.prevTransform, float4(objectSpacePosition, 1.0f)).xyz;
    float4 clipSpacePositionNoOffsetLastFrame = mul(float4(worldSpacePositionLastFrame, 1.0f), g_ViewLastFrame.matWorldToClipNoOffset);

    o_uv = texcoord;
    o_position = clipSpacePosition;
    o_curr_position = clipSpacePositionNoOffset;
    o_prev_position = clipSpacePositionNoOffsetLastFrame;
    o_material = geometry.materialIndex;
}

void ps_main(
    in float4 i_position : SV_Position,
    in float4 i_curr_position : CURR_POSITION,
    in float4 i_prev_position : PREV_POSITION,
    in float3 i_normal_vector : NORMAL_VECTOR,
    in float3 i_prev_normal : PREV_NORMAL,
    in float2 i_uv : TEXCOORD, 
    nointerpolation in uint i_material : MATERIAL,
    out float4 jittered_sample : SV_Target0,
    out float3 normal_vector : SV_Target1,
    out float3 prev_normal : SV_Target2,
    out float4 motion_vector : SV_Target3)
{
    MaterialConstants material = t_MaterialConstants[i_material];

    float3 diffuse = material.baseOrDiffuseColor;

    if (material.baseOrDiffuseTextureIndex >= 0)
    {
        Texture2D diffuseTexture = t_BindlessTextures[material.baseOrDiffuseTextureIndex];

        float4 diffuseTextureValue = diffuseTexture.Sample(s_MaterialSampler, i_uv);
        
        if (material.domain == MaterialDomain_AlphaTested)
            clip(diffuseTextureValue.a - material.alphaCutoff);

        diffuse *= diffuseTextureValue.rgb;
    }
    
    float3 prev_position_clip = i_prev_position.xyz / i_prev_position.w;
    float2 prev_position_screen = float2(prev_position_clip.x * 0.5f + 0.5f, 0.5f - prev_position_clip.y * 0.5f);
    
    float3 curr_position_clip = i_curr_position.xyz / i_curr_position.w;
    float2 curr_position_screen = float2(curr_position_clip.x * 0.5f + 0.5f, 0.5f - curr_position_clip.y * 0.5f);
    motion_vector = float4(curr_position_screen - prev_position_screen, curr_position_clip.z - prev_position_clip.z, 1.0f);

    jittered_sample = float4(diffuse, 1.0f);
    normal_vector = normalize(i_normal_vector);
    prev_normal = normalize(i_prev_normal);
}
