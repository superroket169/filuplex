use crate::context::Context;
use std::sync::Arc;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::DescriptorSet; // Artık sadece struct
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sync::GpuFuture;

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
        .expect("Command buffer builder oluşturulamadı");

        Self { ctx, builder }
    }

    pub fn dispatch(
        &mut self,
        pipeline: Arc<ComputePipeline>,
        descriptor_set: Arc<DescriptorSet>,
        workgroups: [u32; 3],
    ) {
        // Güvenli bağlama işlemleri
        self.builder
            .bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap();

        // Dispatch artık unsafe istiyor
        unsafe {
            self.builder.dispatch(workgroups).unwrap();
        }
    }

    pub fn build(self) -> ExecutableGraph {
        // Gereksiz unsafe bloğu kaldırıldı
        let command_buffer = self.builder.build().unwrap();
        ExecutableGraph {
            ctx: self.ctx,
            command_buffer,
        }
    }
}

// ---- Adım 2: Defalarca Çalıştırılabilen Mühürlü Grafik ----
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
