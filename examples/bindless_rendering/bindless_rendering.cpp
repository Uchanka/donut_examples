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

class BindlessRendering : public app::ApplicationBase
{
private:
	std::shared_ptr<vfs::RootFileSystem> m_RootFS;

    nvrhi::CommandListHandle m_CommandList;

    nvrhi::BindingLayoutHandle m_BindingLayoutMotion;
    nvrhi::BindingLayoutHandle m_BindingLayout;
    nvrhi::BindingLayoutHandle m_BindingLayoutBlend;
    nvrhi::BindingLayoutHandle m_BindlessLayout;

    nvrhi::BindingSetHandle m_BindingSetMotion;
    nvrhi::BindingSetHandle m_BindingSet;
    nvrhi::BindingSetHandle m_BindingSetBlend;

    nvrhi::ShaderHandle m_VertexShaderMotion;
    nvrhi::ShaderHandle m_PixelShaderMotion;
    nvrhi::ShaderHandle m_VertexShader;
    nvrhi::ShaderHandle m_PixelShader;
    nvrhi::ShaderHandle m_VertexShaderBlend;
    nvrhi::ShaderHandle m_PixelShaderBlend;

    nvrhi::GraphicsPipelineHandle m_GraphicsPipelineMotion;
    nvrhi::GraphicsPipelineHandle m_GraphicsPipeline;
    nvrhi::GraphicsPipelineHandle m_GraphicsPipelineBlend;

    nvrhi::BufferHandle m_ViewConstants;
    nvrhi::BufferHandle m_ViewConstantsLastFrame;
    
    nvrhi::TextureHandle m_DepthBuffer;
    nvrhi::TextureHandle m_ColorBuffer;
    nvrhi::TextureHandle m_HistoryBuffer;
    nvrhi::TextureHandle m_DitheredCurrentBuffer;
    nvrhi::TextureHandle m_BlendedCurrentBuffer;
    nvrhi::TextureHandle m_MotionVector;
    nvrhi::FramebufferHandle m_Framebuffer;
   
    std::shared_ptr<engine::ShaderFactory> m_ShaderFactory;
    std::unique_ptr<engine::Scene> m_Scene;
    std::shared_ptr<engine::DescriptorTableManager> m_DescriptorTableManager;
    std::unique_ptr<engine::BindingCache> m_BindingCache;

    app::FirstPersonCamera m_Camera;
    engine::PlanarView m_View;
    
    bool m_EnableAnimations = true;
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

        m_VertexShaderMotion = m_ShaderFactory->CreateShader("/shaders/app/motion_vector.hlsl", "vs_main_motion", nullptr, nvrhi::ShaderType::Vertex);
        m_PixelShaderMotion = m_ShaderFactory->CreateShader("/shaders/app/motion_vector.hlsl", "ps_main_motion", nullptr, nvrhi::ShaderType::Pixel);
        m_VertexShader = m_ShaderFactory->CreateShader("/shaders/app/bindless_rendering.hlsl", "vs_main", nullptr, nvrhi::ShaderType::Vertex);
        m_PixelShader = m_ShaderFactory->CreateShader("/shaders/app/bindless_rendering.hlsl", "ps_main", nullptr, nvrhi::ShaderType::Pixel);
        m_VertexShaderBlend = m_ShaderFactory->CreateShader("/shaders/app/taa_blending.hlsl", "vs_main_blending", nullptr, nvrhi::ShaderType::Vertex);
        m_PixelShaderBlend = m_ShaderFactory->CreateShader("/shaders/app/taa_blending.hlsl", "ps_main_blending", nullptr, nvrhi::ShaderType::Pixel);

        nvrhi::BindlessLayoutDesc bindlessLayoutDesc;
        bindlessLayoutDesc.visibility = nvrhi::ShaderType::All;
        bindlessLayoutDesc.firstSlot = 0;
        bindlessLayoutDesc.maxCapacity = 1024;
        bindlessLayoutDesc.registerSpaces = {
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

        m_ViewConstants = GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(PlanarViewConstants), "ViewConstants", engine::c_MaxRenderPassConstantBufferVersions));
        m_ViewConstantsLastFrame = GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(PlanarViewConstants), "ViewConstantsLastFrame", engine::c_MaxRenderPassConstantBufferVersions));
        
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

        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }

    void BackBufferResizing() override
    { 
        m_DepthBuffer = nullptr;
        m_ColorBuffer = nullptr;
        m_HistoryBuffer = nullptr;
        m_DitheredCurrentBuffer = nullptr;
        m_BlendedCurrentBuffer = nullptr;
        m_MotionVector = nullptr;
        m_Framebuffer = nullptr;
        m_GraphicsPipeline = nullptr;
        m_GraphicsPipelineMotion = nullptr;
        m_GraphicsPipelineBlend = nullptr;
        m_BindingCache->Clear();
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        const auto& fbinfo = framebuffer->getFramebufferInfo();

        if (!m_GraphicsPipeline || !m_GraphicsPipelineBlend || !m_GraphicsPipelineMotion)
        {
            nvrhi::TextureDesc textureDesc;
            textureDesc.format = nvrhi::Format::SRGBA8_UNORM;
            textureDesc.isRenderTarget = true;
            textureDesc.initialState = nvrhi::ResourceStates::RenderTarget;
            textureDesc.keepInitialState = true;
            textureDesc.clearValue = nvrhi::Color(0.f);
            textureDesc.useClearValue = true;
            textureDesc.debugName = "ColorBuffer";
            textureDesc.width = fbinfo.width;
            textureDesc.height = fbinfo.height;
            textureDesc.dimension = nvrhi::TextureDimension::Texture2D;
            m_ColorBuffer = GetDevice()->createTexture(textureDesc);

            textureDesc.format = nvrhi::Format::D24S8;
            textureDesc.debugName = "DepthBuffer";
            textureDesc.initialState = nvrhi::ResourceStates::DepthWrite;
            m_DepthBuffer = GetDevice()->createTexture(textureDesc);

            textureDesc.clearValue = nvrhi::Color(0.f);
            textureDesc.isTypeless = false;
            textureDesc.format = nvrhi::Format::RG16_FLOAT;
            textureDesc.initialState = nvrhi::ResourceStates::RenderTarget;
            textureDesc.isUAV = true;
            textureDesc.debugName = "MotionVector";
            m_MotionVector = GetDevice()->createTexture(textureDesc);

            textureDesc.clearValue = nvrhi::Color(0.f);
            textureDesc.isTypeless = false;
            textureDesc.format = nvrhi::Format::SRGBA8_UNORM;
            textureDesc.initialState = nvrhi::ResourceStates::RenderTarget;
            textureDesc.isUAV = true;
            textureDesc.debugName = "HistoryBuffer";
            m_HistoryBuffer = GetDevice()->createTexture(textureDesc);

            textureDesc.clearValue = nvrhi::Color(0.f);
            textureDesc.isTypeless = false;
            textureDesc.format = nvrhi::Format::SRGBA8_UNORM;
            textureDesc.initialState = nvrhi::ResourceStates::RenderTarget;
            textureDesc.isUAV = true;
            textureDesc.debugName = "DitheredCurrentBuffer";
            m_DitheredCurrentBuffer = GetDevice()->createTexture(textureDesc);

            textureDesc.clearValue = nvrhi::Color(0.f);
            textureDesc.isTypeless = false;
            textureDesc.format = nvrhi::Format::SRGBA8_UNORM;
            textureDesc.initialState = nvrhi::ResourceStates::RenderTarget;
            textureDesc.isUAV = true;
            textureDesc.debugName = "BlendedCurrentBuffer";
            m_BlendedCurrentBuffer = GetDevice()->createTexture(textureDesc);

            nvrhi::FramebufferDesc framebufferDesc;
            framebufferDesc.addColorAttachment(m_ColorBuffer, nvrhi::AllSubresources);
            framebufferDesc.addColorAttachment(m_MotionVector, nvrhi::AllSubresources);
            framebufferDesc.addColorAttachment(m_DitheredCurrentBuffer, nvrhi::AllSubresources);
            framebufferDesc.addColorAttachment(m_BlendedCurrentBuffer, nvrhi::AllSubresources);
            framebufferDesc.setDepthAttachment(m_DepthBuffer);
            m_Framebuffer = GetDevice()->createFramebuffer(framebufferDesc);

            nvrhi::BindingSetDesc bindingSetDescMotion;
            bindingSetDescMotion.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_ViewConstants),
                nvrhi::BindingSetItem::ConstantBuffer(1, m_ViewConstantsLastFrame)
            };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDescMotion, m_BindingLayoutMotion, m_BindingSetMotion);

            nvrhi::GraphicsPipelineDesc pipelineDescMotion;
            pipelineDescMotion.VS = m_VertexShaderMotion;
            pipelineDescMotion.PS = m_PixelShaderMotion;
            pipelineDescMotion.primType = nvrhi::PrimitiveType::TriangleList;
            pipelineDescMotion.bindingLayouts = { m_BindingLayoutMotion };
            pipelineDescMotion.renderState.depthStencilState.depthTestEnable = false;
            pipelineDescMotion.renderState.depthStencilState.stencilEnable = false;
            pipelineDescMotion.renderState.rasterState.setCullNone();
            m_GraphicsPipelineMotion = GetDevice()->createGraphicsPipeline(pipelineDescMotion, m_Framebuffer);

            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_ViewConstants),
                nvrhi::BindingSetItem::PushConstants(1, sizeof(int2)),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_Scene->GetInstanceBuffer()),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_Scene->GetGeometryBuffer()),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_Scene->GetMaterialBuffer()),
                nvrhi::BindingSetItem::Sampler(0, m_CommonPasses->m_AnisotropicWrapSampler)
            };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_BindingLayout, m_BindingSet);
            
            nvrhi::GraphicsPipelineDesc pipelineDesc;
            pipelineDesc.VS = m_VertexShader;
            pipelineDesc.PS = m_PixelShader;
            pipelineDesc.primType = nvrhi::PrimitiveType::TriangleList;
            pipelineDesc.bindingLayouts = { m_BindingLayout, m_BindlessLayout };
            pipelineDesc.renderState.depthStencilState.depthTestEnable = true;
            pipelineDesc.renderState.depthStencilState.depthFunc = nvrhi::ComparisonFunc::GreaterOrEqual;
            pipelineDesc.renderState.rasterState.frontCounterClockwise = true;
            pipelineDesc.renderState.rasterState.setCullBack();
            m_GraphicsPipeline = GetDevice()->createGraphicsPipeline(pipelineDesc, m_Framebuffer);

            nvrhi::BindingSetDesc bindingSetDescBlend;
            bindingSetDescBlend.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_ViewConstants),
                nvrhi::BindingSetItem::PushConstants(1, sizeof(int)),
                nvrhi::BindingSetItem::Texture_SRV(0, m_MotionVector, nvrhi::Format::RG16_FLOAT),
                nvrhi::BindingSetItem::Texture_SRV(1, m_HistoryBuffer, nvrhi::Format::SRGBA8_UNORM),
                nvrhi::BindingSetItem::Texture_SRV(2, m_DitheredCurrentBuffer, nvrhi::Format::SRGBA8_UNORM),
                nvrhi::BindingSetItem::Sampler(0, m_CommonPasses->m_LinearClampSampler)
            };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDescBlend, m_BindingLayoutBlend, m_BindingSetBlend);

            nvrhi::GraphicsPipelineDesc pipelineDescBlend;
            pipelineDescBlend.VS = m_VertexShaderBlend;
            pipelineDescBlend.PS = m_PixelShaderBlend;
            pipelineDescBlend.primType = nvrhi::PrimitiveType::TriangleList;
            pipelineDescBlend.bindingLayouts = { m_BindingLayoutBlend };
            pipelineDescBlend.renderState.depthStencilState.depthTestEnable = false;
            pipelineDescBlend.renderState.depthStencilState.stencilEnable = false;
            pipelineDescBlend.renderState.rasterState.setCullNone();
            m_GraphicsPipelineBlend = GetDevice()->createGraphicsPipeline(pipelineDescBlend, m_Framebuffer);
        }

        m_CommandList->open();
        PlanarViewConstants viewConstants;
        if (GetFrameIndex() != 0)
        {
            m_View.FillPlanarViewConstants(viewConstants);
            m_CommandList->writeBuffer(m_ViewConstantsLastFrame, &viewConstants, sizeof(viewConstants));
        }

        nvrhi::Viewport windowViewport(float(fbinfo.width), float(fbinfo.height));
        m_View.SetViewport(windowViewport);
        m_View.SetMatrices(m_Camera.GetWorldToViewMatrix(), perspProjD3DStyleReverse(dm::PI_f * 0.25f, windowViewport.width() / windowViewport.height(), 0.1f));
        m_View.UpdateCache();
        
        m_Scene->Refresh(m_CommandList, GetFrameIndex());

        m_CommandList->clearDepthStencilTexture(m_DepthBuffer, nvrhi::AllSubresources, true, 0.f, true, 0);

        if (GetFrameIndex() == 0)
        {
            m_View.FillPlanarViewConstants(viewConstants);
            m_CommandList->writeBuffer(m_ViewConstantsLastFrame, &viewConstants, sizeof(viewConstants));
        }

        m_View.FillPlanarViewConstants(viewConstants);
        m_CommandList->writeBuffer(m_ViewConstants, &viewConstants, sizeof(viewConstants));

        nvrhi::GraphicsState stateMotion;
        stateMotion.pipeline = m_GraphicsPipelineMotion;
        stateMotion.framebuffer = m_Framebuffer;
        stateMotion.bindings = { m_BindingSetMotion };
        stateMotion.viewport = m_View.GetViewportState();
        m_CommandList->setGraphicsState(stateMotion);

        nvrhi::DrawArguments argsMotion;
        argsMotion.vertexCount = 6;
        m_CommandList->draw(argsMotion);

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);

        m_CommandList->open();
        m_CommandList->writeBuffer(m_ViewConstants, &viewConstants, sizeof(viewConstants));

        nvrhi::GraphicsState state;
        state.pipeline = m_GraphicsPipeline;
        state.framebuffer = m_Framebuffer;
        state.bindings = { m_BindingSet, m_DescriptorTableManager->GetDescriptorTable() };
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
        m_CommandList->writeBuffer(m_ViewConstants, &viewConstants, sizeof(viewConstants));

        nvrhi::GraphicsState stateBlend;
        stateBlend.pipeline = m_GraphicsPipelineBlend;
        stateBlend.framebuffer = m_Framebuffer;
        stateBlend.bindings = { m_BindingSetBlend };
        stateBlend.viewport = m_View.GetViewportState();
        m_CommandList->setGraphicsState(stateBlend);

        int frameIndex = GetFrameIndex();
        m_CommandList->setPushConstants(&frameIndex, sizeof(frameIndex));

        nvrhi::DrawArguments argsBlend;
        argsBlend.vertexCount = 6;
        m_CommandList->draw(argsBlend);
        m_CommandList->copyTexture(m_HistoryBuffer, nvrhi::TextureSlice(), m_BlendedCurrentBuffer, nvrhi::TextureSlice());

        m_CommonPasses->BlitTexture(m_CommandList, framebuffer, m_ColorBuffer, m_BindingCache.get());
        m_CommandList->clearTextureFloat(m_MotionVector, nvrhi::AllSubresources, nvrhi::Color(0.f));
        m_CommandList->clearTextureFloat(m_ColorBuffer, nvrhi::AllSubresources, nvrhi::Color(0.f));
        m_CommandList->clearTextureFloat(m_DitheredCurrentBuffer, nvrhi::AllSubresources, nvrhi::Color(0.f));
        m_CommandList->clearTextureFloat(m_BlendedCurrentBuffer, nvrhi::AllSubresources, nvrhi::Color(0.f));
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
