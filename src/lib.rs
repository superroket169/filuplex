use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::Queue;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::ComputePipeline;
use vulkano::pipeline::PipelineShaderStageCreateInfo;
use vulkano::pipeline::{Pipeline, PipelineLayout};
use vulkano::shader::ShaderModule;
use vulkano::sync::{self, GpuFuture};
use vulkano::VulkanLibrary;

/*
let ctx = Contex::new(GpuPref::Default);
let operation_sys = BuildInShaders::Multiple::new(/*first_buff_size*/2, /*second_buff_size*/2);
let op = Operation::new(ctx, operation_sys);// like something
let out = op.run([1.0, 2.0], [2.0, 3.0]);
/* out = [2.0, 6.0]*/

*/

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

pub enum GpuPref {
    Default,
    Index(usize),
    Discrete,
}

pub struct Context {
    gpu_pref: GpuPref,
    instance: Arc<Instance>,
    physical_device: Arc<PhysicalDevice>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl Context {
    pub fn new(pref: GpuPref) -> Context {
        let instance = Instance::new(
            VulkanLibrary::new().expect("There's no Vulkan"),
            InstanceCreateInfo::default(),
        )
        .expect("Instance cannot be created");

        let physical_device = instance
            .enumerate_physical_devices()
            .expect("Cannot list GPUs")
            .next()
            .expect("No GPU found");

        println!("GPU: {}", physical_device.properties().device_name);

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .position(|p| p.queue_flags.contains(QueueFlags::COMPUTE))
            .expect("No compute queue found") as u32;

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("Device cannot be created");

        let queue = queues.next().expect("No queue");

        println!("Logical device and queue ready!");

        // buffer sets:
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        Context {
            gpu_pref: pref,
            instance: instance,
            physical_device: physical_device,
            device: device,
            queue: queue,
            memory_allocator: memory_allocator,
            command_buffer_allocator: command_buffer_allocator,
            descriptor_set_allocator: descriptor_set_allocator,
        }
    }
}

// =============================
//            Shaders
// =============================

mod cs_add {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 64) in;

            layout(set = 0, binding = 0) buffer InputA { float a[]; };
            layout(set = 0, binding = 1) buffer InputB { float b[]; };
            layout(set = 0, binding = 2) buffer Output { float c[]; };
            layout(set = 0, binding = 3) uniform Meta { uint length; };

            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i >= length) return;
                c[i] = a[i] + b[i];
            }
        ",
    }
}

mod cs_sub {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 64) in;

            layout(set = 0, binding = 0) buffer InputA { float a[]; };
            layout(set = 0, binding = 1) buffer InputB { float b[]; };
            layout(set = 0, binding = 2) buffer Output { float c[]; };
            layout(set = 0, binding = 3) uniform Meta { uint length; };

            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i >= length) return;
                c[i] = a[i] - b[i];
            }
        ",
    }
}

mod cs_mul {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 64) in;

            layout(set = 0, binding = 0) buffer InputA { float a[]; };
            layout(set = 0, binding = 1) buffer InputB { float b[]; };
            layout(set = 0, binding = 2) buffer Output { float c[]; };
            layout(set = 0, binding = 3) uniform Meta { uint length; };

            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i >= length) return;
                c[i] = a[i] * b[i];
            }
        ",
    }
}

mod cs_div {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 64) in;

            layout(set = 0, binding = 0) buffer InputA { float a[]; };
            layout(set = 0, binding = 1) buffer InputB { float b[]; };
            layout(set = 0, binding = 2) buffer Output { float c[]; };
            layout(set = 0, binding = 3) uniform Meta { uint length; };

            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i >= length) return;
                if (b[i] == 0.0) { c[i] = 0.0; return; }
                c[i] = a[i] / b[i];
            }
        ",
    }
}

mod cs_mod {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 64) in;

            layout(set = 0, binding = 0) buffer InputA { float a[]; };
            layout(set = 0, binding = 1) buffer InputB { float b[]; };
            layout(set = 0, binding = 2) buffer Output { float c[]; };
            layout(set = 0, binding = 3) uniform Meta { uint length; };

            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i >= length) return;
                if (b[i] == 0.0) { c[i] = 0.0; return; }
                c[i] = mod(a[i], b[i]);
            }
        ",
    }
}

mod cs_matris_mul {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 8, local_size_y = 8) in;

            layout(set = 0, binding = 0) buffer MatA { float a[]; };  // M x K
            layout(set = 0, binding = 1) buffer MatB { float b[]; };  // K x N
            layout(set = 0, binding = 2) buffer MatC { float c[]; };  // M x N
            layout(set = 0, binding = 3) uniform Meta { uint M; uint K; uint N; };

            void main() {
                uint row = gl_GlobalInvocationID.x;
                uint col = gl_GlobalInvocationID.y;
                if (row >= M || col >= N) return;

                float sum = 0.0;
                for (uint k = 0; k < K; k++) {
                    sum += a[row * K + k] * b[k * N + col];
                }
                c[row * N + col] = sum;
            }
        ",
    }
}

mod cs_relu {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 64) in;

            layout(set = 0, binding = 0) buffer MatA { float a[]; };
            layout(set = 0, binding = 1) buffer MatB { float b[]; };
            layout(set = 0, binding = 2) uniform Meta { uint length; };

            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i >= length) return;
                b[i] = (a[i] < 0.0) ? 0 : a[i];
            }
        ",
    }
}

/* pub struct BuildInShaders {
    // shader :
    // other somthings
}*/

pub enum BuiltInShaderType {
    Multiply,
    Division,
    Addition,
    Subtraction,
    ModOperation,
    MatrisMul,
    Relu,
}

pub struct BuiltInShader {
    pub shader_type: BuiltInShaderType,
}

// IDEA:
/*
pub enum ShaderSource {
    BuiltIn(BuiltInShaderType),
    Custom(Arc<ShaderModule>),  // kullanıcı kendi yükleyip veriyor
}
*/

impl BuiltInShader {
    pub fn new(shader_type: BuiltInShaderType) -> Self {
        BuiltInShader { shader_type }
    }

    pub fn load(&self, ctx: &Arc<Context>) -> Arc<ShaderModule> {
        match self.shader_type {
            BuiltInShaderType::Addition => cs_add::load(ctx.device.clone()).unwrap(),
            BuiltInShaderType::Multiply => cs_mul::load(ctx.device.clone()).unwrap(),
            BuiltInShaderType::Subtraction => cs_sub::load(ctx.device.clone()).unwrap(),
            BuiltInShaderType::Division => cs_div::load(ctx.device.clone()).unwrap(),
            BuiltInShaderType::ModOperation => cs_mod::load(ctx.device.clone()).unwrap(),
            BuiltInShaderType::MatrisMul => cs_matris_mul::load(ctx.device.clone()).unwrap(),
            BuiltInShaderType::Relu => cs_relu::load(ctx.device.clone()).unwrap(),
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

    pub fn run(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let alloc_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        let buff_info = BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        };

        let input_a = Buffer::from_iter(
            self.context.memory_allocator.clone(),
            buff_info.clone(),
            alloc_info.clone(),
            a.iter().copied(),
        )
        .expect("Buffer cannot be created");

        let length_buf = Buffer::from_data(
            self.context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            alloc_info.clone(),
            a.len() as u32,
        )
        .expect("Length buffer cannot be created");

        let input_b = Buffer::from_iter(
            self.context.memory_allocator.clone(),
            buff_info.clone(),
            alloc_info.clone(),
            b.iter().copied(),
        )
        .expect("Buffer cannot be created");

        let output = Buffer::from_iter(
            self.context.memory_allocator.clone(),
            buff_info.clone(),
            alloc_info.clone(),
            vec![0.0f32; a.len() as usize],
        )
        .expect("Buffer cannot be created");

        let descriptor_set_allocator = &self.context.descriptor_set_allocator;

        let stage_layout = self.pipeline.layout().set_layouts().get(0).unwrap();

        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            stage_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, input_a.clone()),
                WriteDescriptorSet::buffer(1, input_b.clone()),
                WriteDescriptorSet::buffer(2, output.clone()),
                WriteDescriptorSet::buffer(3, length_buf.clone()),
            ],
            [],
        )
        .expect("Descriptor set cannot be created");

        println!("Pipeline and descriptor set ready!");

        // ------- Dispatch --------

        let command_buffer_allocator = &self.context.command_buffer_allocator;

        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let groups = (a.len() as u32 + 63) / 64;

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

        let command_buffer = builder.build().unwrap();

        let future = sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        return output.read().unwrap().to_vec();
    }

    pub fn run_relu(&self, a: &[f32]) -> Vec<f32> {
        let alloc_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        let buff_info = BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        };

        let input_a = Buffer::from_iter(
            self.context.memory_allocator.clone(),
            buff_info.clone(),
            alloc_info.clone(),
            a.iter().copied(),
        )
        .expect("Buffer cannot be created");

        let length_buf = Buffer::from_data(
            self.context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            alloc_info.clone(),
            a.len() as u32,
        )
        .expect("Length buffer cannot be created");

        let output = Buffer::from_iter(
            self.context.memory_allocator.clone(),
            buff_info.clone(),
            alloc_info.clone(),
            vec![0.0f32; a.len() as usize],
        )
        .expect("Buffer cannot be created");

        let descriptor_set_allocator = &self.context.descriptor_set_allocator;

        let stage_layout = self.pipeline.layout().set_layouts().get(0).unwrap();

        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            stage_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, input_a.clone()),
                WriteDescriptorSet::buffer(1, output.clone()),
                WriteDescriptorSet::buffer(2, length_buf.clone()),
            ],
            [],
        )
        .expect("Descriptor set cannot be created");

        println!("Pipeline and descriptor set ready!");

        // ------- Dispatch --------

        let command_buffer_allocator = &self.context.command_buffer_allocator;

        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let groups = (a.len() as u32 + 63) / 64;

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

        let command_buffer = builder.build().unwrap();

        let future = sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        return output.read().unwrap().to_vec();
    }

    pub fn run_matmul(&self, a: &[f32], b: &[f32], m: u32, k: u32, n: u32) -> Vec<f32> {
        let alloc_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        let buff_info = BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        };

        let input_a = Buffer::from_iter(
            self.context.memory_allocator.clone(),
            buff_info.clone(),
            alloc_info.clone(),
            a.iter().copied(),
        )
        .expect("Buffer cannot be created");

        let meta_buf = Buffer::from_data(
            self.context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            alloc_info.clone(),
            MatMeta { m, k, n },
        )
        .expect("Meta buffer cannot be created");

        let input_b = Buffer::from_iter(
            self.context.memory_allocator.clone(),
            buff_info.clone(),
            alloc_info.clone(),
            b.iter().copied(),
        )
        .expect("Buffer cannot be created");

        let output = Buffer::from_iter(
            self.context.memory_allocator.clone(),
            buff_info.clone(),
            alloc_info.clone(),
            vec![0.0f32; (m * n) as usize],
        )
        .expect("Buffer cannot be created");

        // Olması gereken:
        let descriptor_set_allocator = &self.context.descriptor_set_allocator;

        let stage_layout = self.pipeline.layout().set_layouts().get(0).unwrap();

        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            stage_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, input_a.clone()),
                WriteDescriptorSet::buffer(1, input_b.clone()),
                WriteDescriptorSet::buffer(2, output.clone()),
                WriteDescriptorSet::buffer(3, meta_buf),
            ],
            [],
        )
        .expect("Descriptor set cannot be created");

        println!("Pipeline and descriptor set ready!");

        // ------- Dispatch --------

        let command_buffer_allocator = &self.context.command_buffer_allocator;

        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

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

        let command_buffer = builder.build().unwrap();

        let future = sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        return output.read().unwrap().to_vec();
    }
}
