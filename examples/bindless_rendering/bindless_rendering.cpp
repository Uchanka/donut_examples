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

#include <donut/render/GBufferFillPass.h>
#include <donut/render/DrawStrategy.h>
#include <donut/app/ApplicationBase.h>
#include <donut/app/Camera.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/Scene.h>
#include <donut/engine/DescriptorTableManager.h>
#include <donut/engine/BindingCache.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/math/math.h>
#include <nvrhi/utils.h>

using namespace donut;
using namespace donut::math;

#include <donut/shaders/view_cb.h>

static const char* g_WindowTitle = "Donut Example: Bindless Rendering";

static enum AAMode { NATIVE_RESOLUTION, RAW_UPSCALED, TEMPORAL_SUPERSAMPLING, TEMPORAL_ANTIALIASING, PLACE_HOLDER };

class BindlessRendering : public app::ApplicationBase
{
private:
	std::shared_ptr<vfs::RootFileSystem> m_RootFS;

    nvrhi::CommandListHandle m_CommandList;
    
    nvrhi::BindingLayoutHandle m_BindlessLayout;

    nvrhi::BindingLayoutHandle m_RenderBindingLayout;
    nvrhi::BindingLayoutHandle m_MotionBindingLayout;
    nvrhi::BindingLayoutHandle m_UpsampleBindingLayout;
    nvrhi::BindingLayoutHandle m_TSSBindingLayout;
    
    nvrhi::BindingSetHandle m_RenderBindingSet;
    nvrhi::BindingSetHandle m_MotionBindingSet;
    nvrhi::BindingSetHandle m_UpsampleBindingSet;
    nvrhi::BindingSetHandle m_TSSBindingSet;

    nvrhi::ShaderHandle m_RenderVertexShader;
    nvrhi::ShaderHandle m_RenderPixelShader;
    nvrhi::ShaderHandle m_MotionVertexShader;
    nvrhi::ShaderHandle m_MotionPixelShader;
    nvrhi::ShaderHandle m_UpsampleVertexShader;
    nvrhi::ShaderHandle m_UpsamplePixelShader;
    nvrhi::ShaderHandle m_TSSVertexShader;
    nvrhi::ShaderHandle m_TSSPixelShaderPost;

    nvrhi::GraphicsPipelineHandle m_RenderPipeline;
    nvrhi::GraphicsPipelineHandle m_TSSPipeline;

    nvrhi::BufferHandle m_SamplingRate;
    nvrhi::BufferHandle m_FrameIndex;
    nvrhi::BufferHandle m_ThisFrameViewConstants;
    nvrhi::BufferHandle m_LastFrameViewConstants;
    
    //High-res
    nvrhi::TextureHandle m_ColorBuffer;

    nvrhi::TextureHandle m_HistoryColor;
    
    nvrhi::TextureHandle m_SSColorBuffer;
    nvrhi::TextureHandle m_SSNormalBuffer;
    nvrhi::TextureHandle m_SSHistoryNormal;
    nvrhi::TextureHandle m_SSMotionVector;
    
    //Low-res
    nvrhi::TextureHandle m_JitteredColor;
    nvrhi::TextureHandle m_NormalBuffer;
    nvrhi::TextureHandle m_HistoryNormal;
    nvrhi::TextureHandle m_RenderMotionVector;

    //Stencils
    nvrhi::TextureHandle m_DepthBuffer;
    
    nvrhi::FramebufferHandle m_RenderFramebuffer;
    nvrhi::FramebufferHandle m_TSSFramebuffer;
   
    std::shared_ptr<engine::ShaderFactory> m_ShaderFactory;
    std::unique_ptr<engine::Scene> m_Scene;
    std::shared_ptr<engine::DescriptorTableManager> m_DescriptorTableManager;
    std::unique_ptr<engine::BindingCache> m_BindingCache;

    app::FirstPersonCamera m_Camera;
    engine::PlanarView m_View;
    
    bool m_EnableAnimations = true;
    int m_currentAAMode = TEMPORAL_SUPERSAMPLING;
    float m_slidingSamplingRate = 0.5f;
    float m_WallclockTime = 0.f;

public:
    using ApplicationBase::ApplicationBase;

    bool Init()
    {
        std::filesystem::path sceneFileName = app::GetDirectoryWithExecutable().parent_path() / "media/sponza-plus.scene.json";
        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/bindless_rendering" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        
		m_RootFS = std::make_shared<vfs::RootFileSystem>();
		m_RootFS->mount("/shaders/donut", frameworkShaderPath);
		m_RootFS->mount("/shaders/app", appShaderPath);

		m_ShaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), m_RootFS, "/shaders");
		m_CommonPasses = std::make_shared<engine::CommonRenderPasses>(GetDevice(), m_ShaderFactory);
        m_BindingCache = std::make_unique<engine::BindingCache>(GetDevice());

        m_RenderVertexShader = m_ShaderFactory->CreateShader("/shaders/app/bindless_rendering.hlsl", "vs_main", nullptr, nvrhi::ShaderType::Vertex);
        m_RenderPixelShader = m_ShaderFactory->CreateShader("/shaders/app/bindless_rendering.hlsl", "ps_main", nullptr, nvrhi::ShaderType::Pixel);
        m_MotionVertexShader = m_ShaderFactory->CreateShader("/shaders/app/motion_vector.hlsl", "vs_main", nullptr, nvrhi::ShaderType::Vertex);
        m_MotionPixelShader = m_ShaderFactory->CreateShader("/shaders/app/motion_vector.hlsl", "ps_main", nullptr, nvrhi::ShaderType::Pixel);
        m_UpsampleVertexShader = m_ShaderFactory->CreateShader("/shaders/app/upsample.hlsl", "vs_main", nullptr, nvrhi::ShaderType::Vertex);
        m_UpsamplePixelShader = m_ShaderFactory->CreateShader("/shaders/app/upsample.hlsl", "ps_main", nullptr, nvrhi::ShaderType::Pixel);
        m_TSSVertexShader = m_ShaderFactory->CreateShader("/shaders/app/tss.hlsl", "vs_main", nullptr, nvrhi::ShaderType::Vertex);
        m_TSSPixelShaderPost = m_ShaderFactory->CreateShader("/shaders/app/tss.hlsl", "ps_main", nullptr, nvrhi::ShaderType::Pixel);

        nvrhi::BindlessLayoutDesc bindlessLayoutDesc;
        bindlessLayoutDesc.visibility = nvrhi::ShaderType::All;
        bindlessLayoutDesc.firstSlot = 0;
        bindlessLayoutDesc.maxCapacity = 1024;
        bindlessLayoutDesc.registerSpaces = 
        {
            nvrhi::BindingLayoutItem::RawBuffer_SRV(1),
            nvrhi::BindingLayoutItem::Texture_SRV(2)
        };
        m_BindlessLayout = GetDevice()->createBindlessLayout(bindlessLayoutDesc);

        m_DescriptorTableManager = std::make_shared<engine::DescriptorTableManager>(GetDevice(), m_BindlessLayout);

        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
        m_TextureCache = std::make_shared<engine::TextureCache>(GetDevice(), nativeFS, m_DescriptorTableManager);

        m_CommandList = GetDevice()->createCommandList();
        
        SetAsynchronousLoadingEnabled(false);
        BeginLoadingScene(nativeFS, sceneFileName);

        m_Scene->FinishedLoading(GetFrameIndex());

        m_Camera.LookAt(float3(0.f, 1.8f, 0.f), float3(1.f, 1.8f, 0.f));
        m_Camera.SetMoveSpeed(3.f);

        m_SamplingRate = GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(float), "SamplingRate", engine::c_MaxRenderPassConstantBufferVersions));
        m_FrameIndex = GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(int), "FrameIndex", engine::c_MaxRenderPassConstantBufferVersions));
        m_ThisFrameViewConstants = GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(PlanarViewConstants), "ViewConstants", engine::c_MaxRenderPassConstantBufferVersions));
        m_LastFrameViewConstants = GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(PlanarViewConstants), "ViewConstantsLastFrame", engine::c_MaxRenderPassConstantBufferVersions));
        
        GetDevice()->waitForIdle();

        return true;
    }

    bool LoadScene(std::shared_ptr<vfs::IFileSystem> fs, const std::filesystem::path& sceneFileName) override 
    {
        engine::Scene* scene = new engine::Scene(GetDevice(), *m_ShaderFactory, fs, m_TextureCache, m_DescriptorTableManager, nullptr);
        if (scene->Load(sceneFileName))
        {
            m_Scene = std::unique_ptr<engine::Scene>(scene);
            return true;
        }

        return false;
    }

    bool KeyboardUpdate(int key, int scancode, int action, int mods) override
    {
        m_Camera.KeyboardUpdate(key, scancode, action, mods);

        if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        {
            m_EnableAnimations = !m_EnableAnimations;
            return true;
        }
        if (key == GLFW_KEY_T && action == GLFW_PRESS)
        {
            m_currentAAMode = (m_currentAAMode + 1) % PLACE_HOLDER;
            BackBufferResizing();
            return true;
        }

        return true;
    }

    bool MousePosUpdate(double xpos, double ypos) override
    {
        m_Camera.MousePosUpdate(xpos, ypos);
        return true;
    }

    bool MouseButtonUpdate(int button, int action, int mods) override
    {
        m_Camera.MouseButtonUpdate(button, action, mods);
        return true;
    }

    void Animate(float fElapsedTimeSeconds) override
    {
        m_Camera.Animate(fElapsedTimeSeconds);

        if (IsSceneLoaded() && m_EnableAnimations)
        {
            m_WallclockTime += fElapsedTimeSeconds;
            float offset = 0;

            for (const auto& anim : m_Scene->GetSceneGraph()->GetAnimations())
            {
                float duration = anim->GetDuration();
                float integral;
                float animationTime = std::modf((m_WallclockTime + offset) / duration, &integral) * duration;
                (void)anim->Apply(animationTime);
                offset += 1.0f;
            }
        }

        std::string extraInfoOnAAMode = "Current AA Mode: ";
        switch (m_currentAAMode)
        {
        case NATIVE_RESOLUTION:
            extraInfoOnAAMode += " NATIVE";
            break;
        case RAW_UPSCALED:
            extraInfoOnAAMode += " UPSCALED";
            break;
        case TEMPORAL_SUPERSAMPLING:
            extraInfoOnAAMode += " TSS";
            break;
        case TEMPORAL_ANTIALIASING:
            extraInfoOnAAMode += " TAA";
            break;
        default:
            break;
        }

        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle, extraInfoOnAAMode.c_str());
    }

    void BackBufferResizing() override
    { 
        m_DepthBuffer = nullptr;
        m_ColorBuffer = nullptr;
        m_SSColorBuffer = nullptr;
        m_HistoryColor = nullptr;
        m_HistoryNormal = nullptr;
        m_JitteredColor = nullptr;
        m_RenderMotionVector = nullptr;
        m_SSMotionVector = nullptr;
        m_SSNormalBuffer = nullptr;
        m_SSHistoryNormal = nullptr;
        m_NormalBuffer = nullptr;

        m_TSSFramebuffer = nullptr;
        m_RenderFramebuffer = nullptr;

        m_RenderBindingLayout = nullptr;
        m_MotionBindingLayout = nullptr;
        m_UpsampleBindingLayout = nullptr;
        m_TSSBindingLayout = nullptr;

        m_RenderBindingSet = nullptr;
        m_MotionBindingSet = nullptr;
        m_UpsampleBindingSet = nullptr;
        m_TSSBindingSet = nullptr;

        m_RenderPipeline = nullptr;
        m_TSSPipeline = nullptr;

        m_BindingCache->Clear();
    }

    float VanDerCorputSequence(int n, int base)
    {
        float q = 0, bk = (float) 1 / base;

        while (n > 0)
        {
            q += (n % base) * bk;
            n /= base;
            bk /= base;
        }

        return q;
    }

    float2 GetCurrentFramePixelOffset(const int frameIndex)
    {
        float2 fixedMSAA4XPosition[4] = 
        { 
            float2(-0.25f, -0.25f), float2(-0.25f, 0.25f), float2(0.25f, -0.25f), float2(0.25f, 0.25f)
        };
        float2 fixedMSAA16XPosition[16] = 
        { 
            float2(-0.375f, -0.375f), float2(-0.125f, -0.375f), float2(0.125f, -0.375f), float2(0.375f, -0.375f),
            float2(-0.375f, -0.125f), float2(-0.125f, -0.125f), float2(0.125f, -0.125f), float2(0.375f, -0.125f),
            float2(-0.375f, 0.125f), float2(-0.125f, 0.125f), float2(0.125f, 0.125f), float2(0.375f, 0.125f),
            float2(-0.375f, 0.375f), float2(-0.125f, 0.375f), float2(0.125f, 0.375f), float2(0.375f, 0.375f)
        };

        int clampedIndex = frameIndex % 16 + 1;
        switch (m_currentAAMode)
        {
        /*case NATIVE_RESOLUTION:
            return float2(.0f);
        case UPSAMPLED:
            return float2(.0f);*/
        case TEMPORAL_SUPERSAMPLING:
            //return fixedMSAA16XPosition[frameIndex % (sizeof(fixedMSAA16XPosition) / sizeof(fixedMSAA16XPosition[0]))];
            //return fixedMSAA4XPosition[frameIndex % (sizeof(fixedMSAA4XPosition) / sizeof(fixedMSAA4XPosition[0]))];
            return float2(VanDerCorputSequence(clampedIndex, 2) - 0.5f, VanDerCorputSequence(clampedIndex, 3) - 0.5f);
        case TEMPORAL_ANTIALIASING:
            return float2(VanDerCorputSequence(clampedIndex, 2) - 0.5f, VanDerCorputSequence(clampedIndex, 3) - 0.5f);
        default:
            return float2(.0f);
        }
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        const auto& fbinfo = framebuffer->getFramebufferInfo();
        uint32_t upsampledWidth = fbinfo.width;
        uint32_t upsampledHeight = fbinfo.height;
        uint32_t renderWidth = upsampledWidth;
        uint32_t renderHeight = upsampledHeight;
        if (m_currentAAMode != NATIVE_RESOLUTION)
        {
            renderWidth *= m_slidingSamplingRate;
            renderHeight *= m_slidingSamplingRate;
        }

        int frameHasBeenReset = 0;
        if (!m_RenderPipeline || !m_TSSPipeline)
        {
            frameHasBeenReset = 1;
            //High-res texture
            nvrhi::TextureDesc textureDescHighRes;
            textureDescHighRes.format = nvrhi::Format::SRGBA8_UNORM;
            textureDescHighRes.isRenderTarget = true;
            textureDescHighRes.initialState = nvrhi::ResourceStates::RenderTarget;
            textureDescHighRes.keepInitialState = true;
            textureDescHighRes.clearValue = nvrhi::Color(0.f);
            textureDescHighRes.useClearValue = true;
            textureDescHighRes.debugName = "ScreenContent";
            textureDescHighRes.width = upsampledWidth;
            textureDescHighRes.height = upsampledHeight;
            textureDescHighRes.dimension = nvrhi::TextureDimension::Texture2D;
            m_ColorBuffer = GetDevice()->createTexture(textureDescHighRes);

            textureDescHighRes.isTypeless = false;
            textureDescHighRes.format = nvrhi::Format::SRGBA8_UNORM;
            textureDescHighRes.isUAV = true;
            textureDescHighRes.debugName = "SupersampledColor";
            m_SSColorBuffer = GetDevice()->createTexture(textureDescHighRes);

            textureDescHighRes.debugName = "HistoryColor";
            m_HistoryColor = GetDevice()->createTexture(textureDescHighRes);

            textureDescHighRes.debugName = "SupersampledNormalBuffer";
            m_SSNormalBuffer = GetDevice()->createTexture(textureDescHighRes);

            textureDescHighRes.debugName = "SupersampledHistoryNormal";
            m_SSHistoryNormal = GetDevice()->createTexture(textureDescHighRes);

            textureDescHighRes.format = nvrhi::Format::RGBA16_FLOAT;
            textureDescHighRes.debugName = "SupersampledMotionVector";
            m_SSMotionVector = GetDevice()->createTexture(textureDescHighRes);

            //Low-res texture
            nvrhi::TextureDesc textureDescLowRes;
            textureDescLowRes.format = nvrhi::Format::SRGBA8_UNORM;
            textureDescLowRes.isRenderTarget = true;
            textureDescLowRes.initialState = nvrhi::ResourceStates::RenderTarget;
            textureDescLowRes.keepInitialState = true;
            textureDescLowRes.clearValue = nvrhi::Color(0.f);
            textureDescLowRes.useClearValue = true;
            textureDescLowRes.debugName = "JitteredCurrentBuffer";
            textureDescLowRes.width = renderWidth;
            textureDescLowRes.height = renderHeight;
            m_JitteredColor = GetDevice()->createTexture(textureDescLowRes);

            textureDescLowRes.format = nvrhi::Format::D24S8;
            textureDescLowRes.debugName = "DepthBuffer";
            textureDescLowRes.initialState = nvrhi::ResourceStates::DepthWrite;
            m_DepthBuffer = GetDevice()->createTexture(textureDescLowRes);

            textureDescLowRes.isTypeless = false;
            textureDescLowRes.format = nvrhi::Format::SRGBA8_UNORM;
            textureDescLowRes.isUAV = true;
            textureDescLowRes.initialState = nvrhi::ResourceStates::RenderTarget;
            textureDescLowRes.debugName = "NormalBuffer";
            m_NormalBuffer = GetDevice()->createTexture(textureDescLowRes);

            textureDescLowRes.debugName = "HistoryNormal";
            m_HistoryNormal = GetDevice()->createTexture(textureDescLowRes);

            textureDescLowRes.format = nvrhi::Format::RGBA16_FLOAT;
            textureDescLowRes.debugName = "MotionVector";
            m_RenderMotionVector = GetDevice()->createTexture(textureDescLowRes);

            //High-res
            nvrhi::FramebufferDesc framebufferDescHigher;
            framebufferDescHigher.addColorAttachment(m_ColorBuffer, nvrhi::AllSubresources);
            framebufferDescHigher.addColorAttachment(m_SSColorBuffer, nvrhi::AllSubresources);
            framebufferDescHigher.addColorAttachment(m_SSMotionVector, nvrhi::AllSubresources);
            framebufferDescHigher.addColorAttachment(m_SSNormalBuffer, nvrhi::AllSubresources);
            framebufferDescHigher.addColorAttachment(m_SSHistoryNormal, nvrhi::AllSubresources);
            framebufferDescHigher.addColorAttachment(m_HistoryColor, nvrhi::AllSubresources);
            m_TSSFramebuffer = GetDevice()->createFramebuffer(framebufferDescHigher);

            //Low-res
            nvrhi::FramebufferDesc framebufferDescLower;
            framebufferDescLower.addColorAttachment(m_JitteredColor, nvrhi::AllSubresources);
            framebufferDescLower.addColorAttachment(m_NormalBuffer, nvrhi::AllSubresources);
            framebufferDescLower.addColorAttachment(m_HistoryNormal, nvrhi::AllSubresources);
            framebufferDescLower.addColorAttachment(m_RenderMotionVector, nvrhi::AllSubresources);
            framebufferDescLower.setDepthAttachment(m_DepthBuffer);
            m_RenderFramebuffer = GetDevice()->createFramebuffer(framebufferDescLower);

            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_ThisFrameViewConstants),
                nvrhi::BindingSetItem::ConstantBuffer(1, m_LastFrameViewConstants),
                nvrhi::BindingSetItem::PushConstants(2, sizeof(int2)),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_Scene->GetInstanceBuffer()),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_Scene->GetGeometryBuffer()),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_Scene->GetMaterialBuffer()),
                nvrhi::BindingSetItem::Sampler(0, m_CommonPasses->m_AnisotropicWrapSampler)
            };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_RenderBindingLayout, m_RenderBindingSet);
            
            nvrhi::GraphicsPipelineDesc pipelineDesc;
            pipelineDesc.VS = m_RenderVertexShader;
            pipelineDesc.PS = m_RenderPixelShader;
            pipelineDesc.primType = nvrhi::PrimitiveType::TriangleList;
            pipelineDesc.bindingLayouts = { m_RenderBindingLayout, m_BindlessLayout };
            pipelineDesc.renderState.depthStencilState.depthTestEnable = true;
            pipelineDesc.renderState.depthStencilState.depthFunc = nvrhi::ComparisonFunc::GreaterOrEqual;
            pipelineDesc.renderState.rasterState.frontCounterClockwise = true;
            pipelineDesc.renderState.rasterState.setCullBack();

            m_RenderPipeline = GetDevice()->createGraphicsPipeline(pipelineDesc, m_RenderFramebuffer);

            nvrhi::BindingSetDesc bindingSetDescPost;
            bindingSetDescPost.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_ThisFrameViewConstants),
                nvrhi::BindingSetItem::ConstantBuffer(1, m_SamplingRate),
                nvrhi::BindingSetItem::PushConstants(2, sizeof(int2)),
                nvrhi::BindingSetItem::Texture_SRV(0, m_RenderMotionVector, nvrhi::Format::RGBA16_FLOAT),
                nvrhi::BindingSetItem::Texture_SRV(1, m_HistoryColor, nvrhi::Format::SRGBA8_UNORM),
                nvrhi::BindingSetItem::Texture_SRV(2, m_JitteredColor, nvrhi::Format::SRGBA8_UNORM),
                nvrhi::BindingSetItem::Texture_SRV(3, m_NormalBuffer, nvrhi::Format::SRGBA8_UNORM),
                nvrhi::BindingSetItem::Texture_SRV(4, m_HistoryNormal, nvrhi::Format::SRGBA8_UNORM),
                nvrhi::BindingSetItem::Sampler(0, m_CommonPasses->m_LinearClampSampler)
            };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDescPost, m_TSSBindingLayout, m_TSSBindingSet);

            nvrhi::GraphicsPipelineDesc pipelineDescPost;
            pipelineDescPost.VS = m_TSSVertexShader;
            pipelineDescPost.PS = m_TSSPixelShaderPost;
            pipelineDescPost.primType = nvrhi::PrimitiveType::TriangleList;
            pipelineDescPost.bindingLayouts = { m_TSSBindingLayout };
            pipelineDescPost.renderState.depthStencilState.depthTestEnable = false;
            pipelineDescPost.renderState.depthStencilState.stencilEnable = false;
            pipelineDescPost.renderState.rasterState.setCullNone();

            m_TSSPipeline = GetDevice()->createGraphicsPipeline(pipelineDescPost, m_TSSFramebuffer);
        }

        m_CommandList->open();
        
        PlanarViewConstants viewConstants;
        if (GetFrameIndex() != 0)
        {
            m_View.FillPlanarViewConstants(viewConstants);
            m_CommandList->writeBuffer(m_LastFrameViewConstants, &viewConstants, sizeof(viewConstants));
        }

        if (m_currentAAMode == TEMPORAL_SUPERSAMPLING || m_currentAAMode == TEMPORAL_ANTIALIASING)
        {
            m_View.SetPixelOffset(GetCurrentFramePixelOffset(GetFrameIndex()));
        }
        nvrhi::Viewport windowViewport(static_cast<float>(renderWidth), static_cast<float>(renderHeight));
        m_View.SetViewport(windowViewport);
        m_View.SetMatrices(m_Camera.GetWorldToViewMatrix(), perspProjD3DStyleReverse(dm::PI_f * 0.25f, windowViewport.width() / windowViewport.height(), 0.1f));
        m_View.UpdateCache();
        
        m_Scene->Refresh(m_CommandList, GetFrameIndex());

        m_CommandList->clearDepthStencilTexture(m_DepthBuffer, nvrhi::AllSubresources, true, 0.f, true, 0);

        if (GetFrameIndex() == 0)
        {
            m_View.FillPlanarViewConstants(viewConstants);
            m_CommandList->writeBuffer(m_LastFrameViewConstants, &viewConstants, sizeof(viewConstants));
        }
        m_View.FillPlanarViewConstants(viewConstants);
        m_CommandList->writeBuffer(m_ThisFrameViewConstants, &viewConstants, sizeof(viewConstants));

        nvrhi::GraphicsState state;
        state.pipeline = m_RenderPipeline;
        state.framebuffer = m_RenderFramebuffer;
        state.bindings = { m_RenderBindingSet, m_DescriptorTableManager->GetDescriptorTable() };
        state.viewport = m_View.GetViewportState();
        m_CommandList->setGraphicsState(state);

        for (const auto& instance : m_Scene->GetSceneGraph()->GetMeshInstances())
        {
            const auto& mesh = instance->GetMesh();

            for (size_t i = 0; i < mesh->geometries.size(); i++)
            {
                int2 constants = int2(instance->GetInstanceIndex(), int(i));
                m_CommandList->setPushConstants(&constants, sizeof(constants));

                nvrhi::DrawArguments args;
                args.instanceCount = 1;
                args.vertexCount = mesh->geometries[i]->numIndices;
                m_CommandList->draw(args);
            }
        }

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
        
        m_CommandList->open();
        nvrhi::Viewport windowViewportTSS(static_cast<float>(upsampledWidth), static_cast<float>(upsampledHeight));
        m_View.SetViewport(windowViewportTSS);
        m_View.SetMatrices(m_Camera.GetWorldToViewMatrix(), perspProjD3DStyleReverse(dm::PI_f * 0.25f, windowViewportTSS.width() / windowViewportTSS.height(), 0.1f));
        m_View.UpdateCache();

        m_View.FillPlanarViewConstants(viewConstants);

        m_CommandList->writeBuffer(m_ThisFrameViewConstants, &viewConstants, sizeof(viewConstants));
        m_CommandList->writeBuffer(m_SamplingRate, &m_slidingSamplingRate, sizeof(m_slidingSamplingRate));

        nvrhi::GraphicsState statePost;
        statePost.pipeline = m_TSSPipeline;
        statePost.framebuffer = m_TSSFramebuffer;
        statePost.bindings = { m_TSSBindingSet };
        statePost.viewport = m_View.GetViewportState();
        m_CommandList->setGraphicsState(statePost);

        int2 frameStatus = int2(frameHasBeenReset, m_currentAAMode);
        m_CommandList->setPushConstants(&frameStatus, sizeof(frameStatus));
        
        if (frameHasBeenReset == 1)
        {
            m_CommandList->clearTextureFloat(m_HistoryColor, nvrhi::AllSubresources, nvrhi::Color(0.f));
        }

        nvrhi::DrawArguments argsPost;
        argsPost.vertexCount = 6;
        m_CommandList->draw(argsPost);
        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);

        m_CommandList->open();
        m_CommandList->copyTexture(m_HistoryColor, nvrhi::TextureSlice(), m_SSColorBuffer, nvrhi::TextureSlice());
        
        m_CommonPasses->BlitTexture(m_CommandList, framebuffer, m_ColorBuffer, m_BindingCache.get());
        m_CommandList->clearTextureFloat(m_RenderMotionVector, nvrhi::AllSubresources, nvrhi::Color(0.f));
        m_CommandList->clearTextureFloat(m_SSMotionVector, nvrhi::AllSubresources, nvrhi::Color(0.f));
        m_CommandList->clearTextureFloat(m_ColorBuffer, nvrhi::AllSubresources, nvrhi::Color(0.f));
        m_CommandList->clearTextureFloat(m_JitteredColor, nvrhi::AllSubresources, nvrhi::Color(0.f));
        m_CommandList->clearTextureFloat(m_SSColorBuffer, nvrhi::AllSubresources, nvrhi::Color(0.f));
        m_CommandList->clearTextureFloat(m_NormalBuffer, nvrhi::AllSubresources, nvrhi::Color(0.f));
        m_CommandList->clearTextureFloat(m_HistoryNormal, nvrhi::AllSubresources, nvrhi::Color(0.f));
        
        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
    }
};

#ifdef WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    nvrhi::GraphicsAPI api = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    if (api == nvrhi::GraphicsAPI::D3D11)
    {
        log::error("The Bindless Rendering example does not support D3D11.");
        return 1;
    }

    app::DeviceManager* deviceManager = app::DeviceManager::Create(api);

    app::DeviceCreationParameters deviceParams;
#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true; 
    deviceParams.enableNvrhiValidationLayer = true;
#endif
    deviceParams.vsyncEnabled = true;
    deviceParams.backBufferWidth = 1920;
    deviceParams.backBufferHeight = 1080;

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }
    
    {
        BindlessRendering example(deviceManager);
        if (example.Init())
        {
            deviceManager->AddRenderPassToBack(&example);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&example);
        }
    }
    
    deviceManager->Shutdown();

    delete deviceManager;

    return 0;
}
