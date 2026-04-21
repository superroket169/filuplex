use serde::{Deserialize, Serialize};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::shader::ShaderModule;
use vulkano::sync::{self, GpuFuture};
use vulkano::DeviceSize;

use crate::context::Context;
use crate::shaders::*;

#[derive(Serialize, Deserialize)]
pub enum BuiltInShaderType {
    Multiply,
    Sqrt,
    Division,
    Addition,
    Subtraction,
    ModOperation,
    MatrisMul,
    MatrisMulTransposedB,
    Relu,
    Softmax,
    Transpose,
    Sigmoid,
    SigmoidDerivative,
    MatmulRelu,
    RmsPropUpdate,
}

pub struct BuiltInShader {
    pub shader_type: BuiltInShaderType,
}

impl BuiltInShader {
    pub fn new(shader_type: BuiltInShaderType) -> Self {
        BuiltInShader { shader_type }
    }

    pub fn load(&self, ctx: &Arc<Context>) -> Arc<ShaderModule> {
        match self.shader_type {
            BuiltInShaderType::Addition => cs_add::load(ctx.device.clone()).unwrap(),
            BuiltInShaderType::Multiply => cs_mul::load(ctx.device.clone()).unwrap(),
            BuiltInShaderType::Sqrt => cs_sqrt::load(ctx.device.clone()).unwrap(),
            BuiltInShaderType::Subtraction => cs_sub::load(ctx.device.clone()).unwrap(),
            BuiltInShaderType::Division => cs_div::load(ctx.device.clone()).unwrap(),
            BuiltInShaderType::ModOperation => cs_mod::load(ctx.device.clone()).unwrap(),
            BuiltInShaderType::MatrisMul => cs_matris_mul::load(ctx.device.clone()).unwrap(),
            BuiltInShaderType::MatrisMulTransposedB => {
                cs_matris_mul_transposed_b::load(ctx.device.clone()).unwrap()
            }
            BuiltInShaderType::Relu => cs_relu::load(ctx.device.clone()).unwrap(),
            BuiltInShaderType::Softmax => cs_softmax::load(ctx.device.clone()).unwrap(),
            BuiltInShaderType::Transpose => cs_transpose::load(ctx.device.clone()).unwrap(),
            BuiltInShaderType::Sigmoid => cs_sigmoid::load(ctx.device.clone()).unwrap(),
            BuiltInShaderType::SigmoidDerivative => {
                cs_sigmoid_derivative::load(ctx.device.clone()).unwrap()
            }
            BuiltInShaderType::MatmulRelu => cs_matmul_relu::load(ctx.device.clone()).unwrap(),
            BuiltInShaderType::RmsPropUpdate => {
                cs_rmsprop_update::load(ctx.device.clone()).unwrap()
            }
        }
    }
}

pub struct Operation {
    context: Arc<Context>,
    pipeline: Arc<ComputePipeline>,
}

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct MatMeta {
    m: u32,
    k: u32,
    n: u32,
}

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct RmsPropMeta {
    length: u32,
    learning_rate: f32,
    decay: f32,
    epsilon: f32,
}

impl Operation {
    pub fn new(ctx: Arc<Context>, shader: Arc<ShaderModule>) -> Self {
        let entry_point = shader.entry_point("main").unwrap();
        let stage = PipelineShaderStageCreateInfo::new(entry_point.clone());

        let layout = PipelineLayout::new(
            ctx.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(ctx.device.clone())
                .unwrap(),
        )
        .unwrap();

        let pipeline = ComputePipeline::new(
            ctx.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap();

        Operation {
            context: ctx,
            pipeline,
        }
    }

    // =========================================================================
    //                         Internal helpers
    // =========================================================================
    fn staging_upload(&self, data: &[f32]) -> Subbuffer<[f32]> {
        let buf = Buffer::new_slice::<f32>(
            self.context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.len() as DeviceSize,
        )
        .expect("Staging buffer cannot be created");

        buf.write()
            .expect("Staging buffer cannot be mapped")
            .copy_from_slice(data);

        buf
    }
    fn device_storage(&self, len: usize, extra_usage: BufferUsage) -> Subbuffer<[f32]> {
        Buffer::new_slice::<f32>(
            self.context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | extra_usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            len as DeviceSize,
        )
        .expect("Device storage buffer cannot be created")
    }
    fn upload_input(
        &self,
        builder: &mut AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer>,
        data: &[f32],
    ) -> Subbuffer<[f32]> {
        let staging = self.staging_upload(data);
        let device_buf = self.device_storage(data.len(), BufferUsage::TRANSFER_DST);
        builder
            .copy_buffer(CopyBufferInfo::buffers(staging, device_buf.clone()))
            .expect("copy_buffer failed");
        device_buf
    }
    fn uniform<T>(&self, data: T) -> Subbuffer<T>
    where
        T: vulkano::buffer::BufferContents,
    {
        Buffer::from_data(
            self.context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data,
        )
        .expect("Uniform buffer cannot be created")
    }
    fn readback(
        &self,
        builder: &mut AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer>,
        device_output: Subbuffer<[f32]>,
        len: usize,
    ) -> Subbuffer<[f32]> {
        let host_buf = Buffer::new_slice::<f32>(
            self.context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            len as DeviceSize,
        )
        .expect("Readback buffer cannot be created");

        builder
            .copy_buffer(CopyBufferInfo::buffers(device_output, host_buf.clone()))
            .expect("copy_buffer readback failed");

        host_buf
    }
    fn new_builder(
        &self,
    ) -> AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            self.context.command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("Command buffer builder cannot be created")
    }
    fn submit_and_wait(
        &self,
        builder: AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer>,
    ) {
        let command_buffer = builder.build().expect("Command buffer build failed");
        sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), command_buffer)
            .expect("Execute failed")
            .then_signal_fence_and_flush()
            .expect("Flush failed")
            .wait(None)
            .expect("Fence wait failed");
    }
    pub fn run(&self, a: Subbuffer<[f32]>, b: Subbuffer<[f32]>) -> Subbuffer<[f32]> {
        let mut builder = self.new_builder();

        let output = self.device_storage(
            a.len() as usize,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        );

        let length_buf = self.uniform::<u32>(a.len() as u32);
        let groups = (a.len() as u32 + 63) / 64;
        let stage_layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = DescriptorSet::new(
            self.context.descriptor_set_allocator.clone(),
            stage_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, a),
                WriteDescriptorSet::buffer(1, b),
                WriteDescriptorSet::buffer(2, output.clone()),
                WriteDescriptorSet::buffer(3, length_buf),
            ],
            [],
        )
        .expect("Descriptor set cannot be created");

        unsafe {
            builder
                .bind_pipeline_compute(self.pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Compute,
                    self.pipeline.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .dispatch([groups, 1, 1])
                .unwrap();
        }

        self.submit_and_wait(builder);
        output
    }
    pub fn run_relu(&self, a: Subbuffer<[f32]>) -> Subbuffer<[f32]> {
        let mut builder = self.new_builder();

        let output = self.device_storage(
            a.len() as usize,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        );
        let length_buf = self.uniform::<u32>(a.len() as u32);

        let groups = (a.len() as u32 + 63) / 64;
        let stage_layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = DescriptorSet::new(
            self.context.descriptor_set_allocator.clone(),
            stage_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, a),
                WriteDescriptorSet::buffer(1, output.clone()),
                WriteDescriptorSet::buffer(2, length_buf),
            ],
            [],
        )
        .expect("Descriptor set cannot be created");

        unsafe {
            builder
                .bind_pipeline_compute(self.pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Compute,
                    self.pipeline.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .dispatch([groups, 1, 1])
                .unwrap();
        }

        self.submit_and_wait(builder);

        output
    }
    pub fn run_matmul(
        &self,
        a: Subbuffer<[f32]>,
        b: Subbuffer<[f32]>,
        m: u32,
        k: u32,
        n: u32,
    ) -> Subbuffer<[f32]> {
        let mut builder = self.new_builder();

        let output = self.device_storage((m * n) as usize, BufferUsage::TRANSFER_SRC);
        let meta_buf = self.uniform::<MatMeta>(MatMeta { m, k, n });

        let stage_layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = DescriptorSet::new(
            self.context.descriptor_set_allocator.clone(),
            stage_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, a),
                WriteDescriptorSet::buffer(1, b),
                WriteDescriptorSet::buffer(2, output.clone()),
                WriteDescriptorSet::buffer(3, meta_buf),
            ],
            [],
        )
        .expect("Descriptor set cannot be created");

        unsafe {
            builder
                .bind_pipeline_compute(self.pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Compute,
                    self.pipeline.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .dispatch([(m + 7) / 8, (n + 7) / 8, 1])
                .unwrap();
        }

        self.submit_and_wait(builder);

        output
    }
    pub fn run_fn(&self, a: Subbuffer<[f32]>) -> Subbuffer<[f32]> {
        let mut builder = self.new_builder();

        let length_buf = self.uniform::<u32>(a.len() as u32);

        let output = self.device_storage(
            a.len() as usize,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        );

        let groups = (a.len() as u32 + 63) / 64;
        let stage_layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = DescriptorSet::new(
            self.context.descriptor_set_allocator.clone(),
            stage_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, a),
                WriteDescriptorSet::buffer(1, output.clone()),
                WriteDescriptorSet::buffer(2, length_buf),
            ],
            [],
        )
        .expect("Descriptor set cannot be created");

        unsafe {
            builder
                .bind_pipeline_compute(self.pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Compute,
                    self.pipeline.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .dispatch([groups, 1, 1])
                .unwrap();
        }

        self.submit_and_wait(builder);

        output
    }
    pub fn run_matmul_relu(
        &self,
        a: Subbuffer<[f32]>,
        b: Subbuffer<[f32]>,
        m: u32,
        k: u32,
        n: u32,
    ) -> Subbuffer<[f32]> {
        let mut builder = self.new_builder();

        let output = self.device_storage(
            (m * n) as usize,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        );

        let meta_buf = self.uniform::<MatMeta>(MatMeta { m, k, n });

        let stage_layout = self.pipeline.layout().set_layouts().get(0).unwrap();

        let descriptor_set = DescriptorSet::new(
            self.context.descriptor_set_allocator.clone(),
            stage_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, a),
                WriteDescriptorSet::buffer(1, b),
                WriteDescriptorSet::buffer(2, output.clone()),
                WriteDescriptorSet::buffer(3, meta_buf),
            ],
            [],
        )
        .unwrap();

        let groups_x = (m + 7) / 8;
        let groups_y = (n + 7) / 8;

        unsafe {
            builder
                .bind_pipeline_compute(self.pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Compute,
                    self.pipeline.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .dispatch([groups_x, groups_y, 1])
                .unwrap();
        }

        self.submit_and_wait(builder);

        output
    }
}
