use std::{
    ffi::{CStr, CString},
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use ash::{extensions::khr::PipelineExecutableProperties, vk, Entry};
use flat_ada_a1_candidate::{ada_a1_branchless_wgsl, ADA_A1_FWD_WGSL};
use flat_attention::FLAT_FWD_WGSL;
use naga::valid::{Capabilities, ValidationFlags, Validator};

struct Variant {
    name: &'static str,
    entry_point: &'static str,
    source: String,
}

fn git_sha() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|sha| sha.trim().to_owned())
        .filter(|sha| !sha.is_empty())
        .unwrap_or_else(|| "unknown".to_owned())
}

fn c_array_text(chars: &[std::os::raw::c_char]) -> String {
    unsafe { CStr::from_ptr(chars.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn one_line(value: &str) -> String {
    value.replace(['\n', '\r'], " ")
}

fn file_slug(value: &str) -> String {
    let mut slug = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
            slug.push(ch.to_ascii_lowercase());
        } else if !slug.ends_with('_') {
            slug.push('_');
        }
    }
    let trimmed = slug.trim_matches('_');
    if trimmed.is_empty() {
        "representation".to_owned()
    } else {
        trimmed.to_owned()
    }
}

fn spv_words(source: &str, entry_point: &str) -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("WGSL parse failed for {entry_point}: {error:?}"));
    let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|error| panic!("WGSL validation failed for {entry_point}: {error:?}"));
    let options = naga::back::spv::Options::default();
    let pipeline_options = naga::back::spv::PipelineOptions {
        entry_point: entry_point.to_owned(),
        shader_stage: naga::ShaderStage::Compute,
    };
    naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline_options))
        .unwrap_or_else(|error| panic!("SPIR-V generation failed for {entry_point}: {error:?}"))
}

fn write_spv(path: &Path, words: &[u32]) {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for &word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    fs::write(path, bytes).unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
}

fn statistic_value(statistic: &vk::PipelineExecutableStatisticKHR) -> String {
    unsafe {
        match statistic.format.as_raw() {
            0 => format!("{}", statistic.value.b32 != vk::FALSE),
            1 => statistic.value.i64.to_string(),
            2 => statistic.value.u64.to_string(),
            3 => format!("{:.9}", statistic.value.f64),
            other => format!("unsupported-format-{other}"),
        }
    }
}

fn has_extension(properties: &[vk::ExtensionProperties], name: &CStr) -> bool {
    properties.iter().any(|property| {
        unsafe { CStr::from_ptr(property.extension_name.as_ptr()) } == name
    })
}

fn make_descriptor_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
    let mut bindings = Vec::with_capacity(5);
    for binding in 0..4 {
        bindings.push(
            vk::DescriptorSetLayoutBinding::builder()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
        );
    }
    bindings.push(
        vk::DescriptorSetLayoutBinding::builder()
            .binding(4)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build(),
    );
    let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&create_info, None) }
        .expect("descriptor set layout creation failed")
}

fn create_capture_pipeline(
    device: &ash::Device,
    pipeline_layout: vk::PipelineLayout,
    words: &[u32],
    entry_point: &str,
) -> vk::Pipeline {
    let module_info = vk::ShaderModuleCreateInfo::builder().code(words);
    let shader_module = unsafe { device.create_shader_module(&module_info, None) }
        .unwrap_or_else(|error| panic!("shader module creation failed for {entry_point}: {error:?}"));
    let entry_name = CString::new(entry_point).expect("entry point contains no NUL");
    let stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&entry_name)
        .build();
    let flags = vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR
        | vk::PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS_KHR;
    let create_info = vk::ComputePipelineCreateInfo::builder()
        .flags(flags)
        .stage(stage)
        .layout(pipeline_layout)
        .build();
    let pipeline = match unsafe {
        device.create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
    } {
        Ok(pipelines) => pipelines[0],
        Err((_, error)) => panic!("compute pipeline creation failed for {entry_point}: {error:?}"),
    };
    unsafe { device.destroy_shader_module(shader_module, None) };
    pipeline
}

fn write_internal_representations(
    extension: &PipelineExecutableProperties,
    executable_info: &vk::PipelineExecutableInfoKHR,
    variant_dir: &Path,
    executable_index: usize,
) -> usize {
    let mut representations = unsafe {
        extension.get_pipeline_executable_internal_representations(executable_info)
    }
    .expect("internal representation metadata query failed");
    if representations.is_empty() {
        return 0;
    }

    let mut storage: Vec<Vec<u8>> = representations
        .iter()
        .map(|representation| vec![0_u8; representation.data_size])
        .collect();
    for (representation, bytes) in representations.iter_mut().zip(storage.iter_mut()) {
        representation.data_size = bytes.len();
        representation.p_data = bytes.as_mut_ptr().cast();
    }

    let mut count = u32::try_from(representations.len()).expect("representation count fits u32");
    let result = unsafe {
        (extension
            .fp()
            .get_pipeline_executable_internal_representations_khr)(
            extension.device(),
            executable_info,
            &mut count,
            representations.as_mut_ptr(),
        )
    };
    assert_eq!(
        result,
        vk::Result::SUCCESS,
        "internal representation payload query failed"
    );

    for (representation_index, (representation, bytes)) in
        representations.iter().zip(storage.iter()).enumerate()
    {
        let name = c_array_text(&representation.name);
        let description = c_array_text(&representation.description);
        let payload_len = representation.data_size.min(bytes.len());
        let payload = &bytes[..payload_len];
        let extension_name = if representation.is_text == vk::TRUE {
            "txt"
        } else {
            "bin"
        };
        let file_name = format!(
            "executable-{executable_index:02}-representation-{representation_index:02}-{}.{}",
            file_slug(&name),
            extension_name
        );
        let path = variant_dir.join(file_name);
        if representation.is_text == vk::TRUE {
            let text_len = payload.iter().position(|&byte| byte == 0).unwrap_or(payload.len());
            fs::write(&path, &payload[..text_len]).unwrap_or_else(|error| {
                panic!("failed to write {}: {error}", path.display())
            });
        } else {
            fs::write(&path, payload).unwrap_or_else(|error| {
                panic!("failed to write {}: {error}", path.display())
            });
        }
        println!(
            "internal_representation variant={} executable={} index={} name={} is_text={} bytes={} path={} description={}",
            variant_dir.file_name().unwrap_or_default().to_string_lossy(),
            executable_index,
            representation_index,
            one_line(&name),
            representation.is_text == vk::TRUE,
            payload_len,
            path.display(),
            one_line(&description),
        );
    }
    representations.len()
}

fn inspect_pipeline(
    extension: &PipelineExecutableProperties,
    pipeline: vk::Pipeline,
    variant: &str,
    variant_dir: &Path,
) {
    let pipeline_info = vk::PipelineInfoKHR::builder().pipeline(pipeline).build();
    let executables = unsafe { extension.get_pipeline_executable_properties(&pipeline_info) }
        .unwrap_or_else(|error| panic!("executable property query failed for {variant}: {error:?}"));
    println!("variant={variant} executable_count={}", executables.len());

    for (index, executable) in executables.iter().enumerate() {
        let name = c_array_text(&executable.name);
        let description = c_array_text(&executable.description);
        println!(
            "executable variant={variant} index={index} name={} description={} subgroup_size={} stages_raw={}",
            one_line(&name),
            one_line(&description),
            executable.subgroup_size,
            executable.stages.as_raw(),
        );
        let executable_info = vk::PipelineExecutableInfoKHR::builder()
            .pipeline(pipeline)
            .executable_index(u32::try_from(index).expect("executable index fits u32"))
            .build();
        let statistics = unsafe {
            extension.get_pipeline_executable_statistics(&executable_info)
        }
        .unwrap_or_else(|error| panic!("statistics query failed for {variant}: {error:?}"));
        println!(
            "statistics variant={variant} executable={index} count={}",
            statistics.len()
        );
        for statistic in &statistics {
            println!(
                "stat variant={variant} executable={index} name={} value={} format_raw={} description={}",
                one_line(&c_array_text(&statistic.name)),
                statistic_value(statistic),
                statistic.format.as_raw(),
                one_line(&c_array_text(&statistic.description)),
            );
        }
        let representation_count = write_internal_representations(
            extension,
            &executable_info,
            variant_dir,
            index,
        );
        println!(
            "internal_representation_count variant={variant} executable={index} count={representation_count}"
        );
    }
}

fn run() {
    let sha = git_sha();
    let output_root = std::env::var_os("FLAT_ADA_A1_PROBE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(format!(
                "/tmp/ada-a1-vulkan-pipeline-probe-{}",
                sha.chars().take(12).collect::<String>()
            ))
        });
    fs::create_dir_all(&output_root).unwrap_or_else(|error| {
        panic!("failed to create {}: {error}", output_root.display())
    });

    let variants = [
        Variant {
            name: "q4",
            entry_point: "flat_attention_forward",
            source: FLAT_FWD_WGSL.to_owned(),
        },
        Variant {
            name: "a1_branched",
            entry_point: "flat_attention_forward_ada_a1",
            source: ADA_A1_FWD_WGSL.to_owned(),
        },
        Variant {
            name: "a1b_branchless",
            entry_point: "flat_attention_forward_ada_a1_branchless",
            source: ada_a1_branchless_wgsl(),
        },
    ];
    let compiled: Vec<Vec<u32>> = variants
        .iter()
        .map(|variant| spv_words(&variant.source, variant.entry_point))
        .collect();
    for (variant, words) in variants.iter().zip(compiled.iter()) {
        write_spv(&output_root.join(format!("{}.spv", variant.name)), words);
    }

    let entry = unsafe { Entry::load() }.expect("failed to load Vulkan loader");
    let app_name = CString::new("flat-ada-a1-vulkan-pipeline-probe").unwrap();
    let app_info = vk::ApplicationInfo::builder()
        .application_name(&app_name)
        .application_version(1)
        .engine_name(&app_name)
        .engine_version(1)
        .api_version(vk::API_VERSION_1_1);
    let instance_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_info, None) }
        .expect("Vulkan instance creation failed");

    let physical_devices = unsafe { instance.enumerate_physical_devices() }
        .expect("physical device enumeration failed");
    let (physical_device, properties) = physical_devices
        .into_iter()
        .find_map(|physical_device| {
            let properties = unsafe { instance.get_physical_device_properties(physical_device) };
            let name = c_array_text(&properties.device_name);
            (name.contains("NVIDIA") && name.contains("Thor"))
                .then_some((physical_device, properties))
        })
        .expect("NVIDIA Thor Vulkan physical device not found");
    let device_name = c_array_text(&properties.device_name);

    let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device) }
        .expect("device extension enumeration failed");
    assert!(
        has_extension(&extensions, PipelineExecutableProperties::name()),
        "VK_KHR_pipeline_executable_properties is not available"
    );

    let mut queried_executable_features =
        vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default();
    let mut queried_features2 = vk::PhysicalDeviceFeatures2::builder()
        .push_next(&mut queried_executable_features)
        .build();
    unsafe { instance.get_physical_device_features2(physical_device, &mut queried_features2) };
    assert_eq!(
        queried_executable_features.pipeline_executable_info,
        vk::TRUE,
        "pipelineExecutableInfo feature is not enabled by the physical device"
    );

    let queue_properties =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let queue_family_index = queue_properties
        .iter()
        .position(|properties| properties.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .and_then(|index| u32::try_from(index).ok())
        .expect("compute-capable queue family not found");
    let priorities = [1.0_f32];
    let queue_info = [vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities)
        .build()];
    let extension_names = [PipelineExecutableProperties::name().as_ptr()];
    let mut enabled_executable_features =
        vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::builder()
            .pipeline_executable_info(true);
    let device_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_info)
        .enabled_extension_names(&extension_names)
        .push_next(&mut enabled_executable_features);
    let device = unsafe { instance.create_device(physical_device, &device_info, None) }
        .expect("Vulkan device creation failed");
    let executable_extension = PipelineExecutableProperties::new(&instance, &device);

    let descriptor_set_layout = make_descriptor_set_layout(&device);
    let set_layouts = [descriptor_set_layout];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
        .expect("pipeline layout creation failed");

    println!("probe=ada_a1_vulkan_pipeline_executable_properties");
    println!("git_sha={sha}");
    println!("device_name={device_name}");
    println!("api_version={}.{}.{}", vk::api_version_major(properties.api_version), vk::api_version_minor(properties.api_version), vk::api_version_patch(properties.api_version));
    println!("driver_version_raw={}", properties.driver_version);
    println!("pipeline_executable_info=true");
    println!("capture_statistics=true");
    println!("capture_internal_representations=true");
    println!("output_dir={}", output_root.display());

    let mut pipelines = Vec::with_capacity(variants.len());
    for ((variant, words), _) in variants.iter().zip(compiled.iter()).zip(0..) {
        let variant_dir = output_root.join(variant.name);
        fs::create_dir_all(&variant_dir).unwrap_or_else(|error| {
            panic!("failed to create {}: {error}", variant_dir.display())
        });
        let pipeline = create_capture_pipeline(&device, pipeline_layout, words, variant.entry_point);
        inspect_pipeline(&executable_extension, pipeline, variant.name, &variant_dir);
        pipelines.push(pipeline);
    }

    unsafe {
        for pipeline in pipelines {
            device.destroy_pipeline(pipeline, None);
        }
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_set_layout(descriptor_set_layout, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    println!("probe_status=complete");
}

fn main() {
    run();
}
