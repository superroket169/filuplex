use std::sync::Arc;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::VulkanLibrary;

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

        let physical_devices: Vec<_> = instance
            .enumerate_physical_devices()
            .expect("Cannot list GPUs")
            .collect();

        assert!(!physical_devices.is_empty(), "No GPU found");

        let physical_device = match &pref {
            GpuPref::Default => physical_devices[0].clone(),
            GpuPref::Index(i) => physical_devices
                .get(*i)
                .unwrap_or_else(|| panic!("GPU index {} out of range", i))
                .clone(),
            GpuPref::Discrete => physical_devices
                .iter()
                .find(|d| {
                    d.properties().device_type
                        == vulkano::device::physical::PhysicalDeviceType::DiscreteGpu
                })
                .unwrap_or(&physical_devices[0])
                .clone(),
        };

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
