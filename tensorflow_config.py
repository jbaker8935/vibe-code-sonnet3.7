"""TensorFlow configuration utilities for optimal performance."""
import os
import tensorflow as tf

def configure_tensorflow():
    """Configure TensorFlow settings for optimal performance."""
    # Set memory growth and optimization flags before any other TF operations
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '1'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
    
    # Use float32 for better numerical stability
    tf.keras.backend.set_floatx('float32')
    
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Enable memory growth
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                
            # Set memory limit only, removed preallocate option
            for device in physical_devices:
                tf.config.set_logical_device_configuration(
                    device,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=3 * 1024)]  # 3GB limit
                )
            
            # Enable XLA and other optimizations
            tf.config.optimizer.set_jit(True)
            tf.config.optimizer.set_experimental_options({
                'layout_optimizer': True,
                'constant_folding': True,
                'shape_optimization': True,
                'remapping': True,
                'arithmetic_optimization': True,
                'dependency_optimization': True,
                'loop_optimization': True,
                'function_optimization': True,
                'debug_stripper': True,
            })
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            print("Falling back to CPU...")
            
    # Optimize for CPU if no GPU
    else:
        tf.config.optimizer.set_jit(True)
        tf.config.threading.set_inter_op_parallelism_threads(4)
        tf.config.threading.set_intra_op_parallelism_threads(4)