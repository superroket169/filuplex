use crate::context::Context;
use crate::ops::GpuBuffer;
use std::sync::Arc;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::shader::ShaderModule;
use vulkano::sync::GpuFuture;

pub struct ExecutableGraph {
    ctx: Arc<Context>,
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
}

impl ExecutableGraph {
    pub fn execute(&self) {
        vulkano::sync::now(self.ctx.device.clone())
            .then_execute(self.ctx.queue.clone(), self.command_buffer.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .expect("GPU hesaplaması çöktü!");
    }
}

pub struct ComputeGraphBuilder {
    ctx: Arc<Context>,
    builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
}

impl ComputeGraphBuilder {
    pub fn new(ctx: Arc<Context>) -> Self {
        let builder = AutoCommandBufferBuilder::primary(
            ctx.command_buffer_allocator.clone(),
            ctx.queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();

        Self { ctx, builder }
    }

    pub fn add_operation(
        &mut self,
        shader: Arc<ShaderModule>,
        buffers: Vec<(u32, &GpuBuffer)>,
        workgroups: [u32; 3],
    ) {
        let mut writes = Vec::new();
        for (binding_index, gpu_buf) in buffers.into_iter() {
            writes.push(WriteDescriptorSet::buffer(
                binding_index,
                gpu_buf.inner.clone(),
            ));
        }

        let entry = shader.entry_point("main").unwrap();
        let stage = vulkano::pipeline::PipelineShaderStageCreateInfo::new(entry);

        let layout = vulkano::pipeline::PipelineLayout::new(
            self.ctx.device.clone(),
            vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(self.ctx.device.clone())
                .unwrap(),
        )
        .unwrap();

        let pipeline = vulkano::pipeline::ComputePipeline::new(
            self.ctx.device.clone(),
            None,
            vulkano::pipeline::compute::ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap();

        let desc_layout = pipeline.layout().set_layouts().first().unwrap().clone();
        let desc_set = DescriptorSet::new(
            self.ctx.descriptor_set_allocator.clone(),
            desc_layout,
            writes,
            [],
        )
        .expect("Descriptor Set cannot be created");

        self.builder
            .bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                desc_set,
            )
            .unwrap();

        unsafe {
            self.builder.dispatch(workgroups).unwrap();
        }
    }

    pub fn build(self) -> ExecutableGraph {
        let command_buffer = self.builder.build().unwrap();
        ExecutableGraph {
            ctx: self.ctx,
            command_buffer,
        }
    }
}
