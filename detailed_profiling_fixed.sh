#!/bin/bash

mkdir -p profiling_results

if command -v nvidia-smi &> /dev/null; then
    nvprof --log-file "profiling_results/nvprof_output_%p.txt" ./mnist_gpu
    nvprof --profile-child-processes --print-gpu-trace --log-file "profiling_results/gpu_memory_trace_%p.txt" ./mnist_gpu
else
    echo "NVIDIA drivers not detected. GPU profiling will be skipped."
fi

./mnist_gpu > profiling_results/gpu_benchmark.txt
./mnist_gpu_nchw > profiling_results/gpu_nchw_benchmark.txt
python3 inference_mnist_tinycnn_cpu.py > profiling_results/cpu_benchmark.txt
python3 inference_mnist_tinycnn_cpu_optimized.py > profiling_results/cpu_opt_benchmark.txt

/usr/bin/time -v ./mnist_gpu > /dev/null 2> profiling_results/gpu_time_profile.txt
/usr/bin/time -v ./mnist_gpu_nchw > /dev/null 2> profiling_results/gpu_nchw_time_profile.txt
/usr/bin/time -v python3 inference_mnist_tinycnn_cpu.py > /dev/null 2> profiling_results/cpu_time_profile.txt
/usr/bin/time -v python3 inference_mnist_tinycnn_cpu_optimized.py > /dev/null 2> profiling_results/cpu_opt_time_profile.txt


for impl in "gpu" "gpu_nchw" "cpu" "cpu_opt"; do
    cat > monitor_memory_$impl.sh << EOF
#!/bin/bash
pid=""
if [ "$impl" = "gpu" ]; then
    ./mnist_gpu &
    pid=\$!
elif [ "$impl" = "gpu_nchw" ]; then
    ./mnist_gpu_nchw &
    pid=\$!
elif [ "$impl" = "cpu" ]; then
    python3 inference_mnist_tinycnn_cpu.py &
    pid=\$!
else
    python3 inference_mnist_tinycnn_cpu_optimized.py &
    pid=\$!
fi

echo "timestamp,rss_kb,vsz_kb" > profiling_results/${impl}_memory.csv
while ps -p \$pid > /dev/null; do
    if [ -e /proc/\$pid/status ]; then
        rss=\$(grep VmRSS /proc/\$pid/status | awk '{print \$2}')
        vsz=\$(grep VmSize /proc/\$pid/status | awk '{print \$2}')
        echo "\$(date +%s),\$rss,\$vsz" >> profiling_results/${impl}_memory.csv
    fi
    sleep 0.05
done
EOF
    chmod +x monitor_memory_$impl.sh
    ./monitor_memory_$impl.sh
done


cat > analyze_results.py << 'EOF'
#!/usr/bin/env python3
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def parse_benchmark_files():
    results = {}
    
    patterns = {
        'gpu': r'GPU Result:.*?Throughput:\s*([\d.]+)\s*img/s.*?Accuracy:\s*([\d.]+)%',
        'gpu_nchw': r'GPU Result \(NCHW\):.*?Throughput:\s*([\d.]+)\s*img/s.*?Accuracy:\s*([\d.]+)%',
        'cpu': r'CPU inference accuracy:\s*([\d.]+)%.*?throughput:\s*([\d.]+)\s*img/s',
        'cpu_opt': r'CPU inference accuracy:\s*([\d.]+)%.*?throughput:\s*([\d.]+)\s*img/s'
    }
    
    for impl in ['gpu', 'gpu_nchw', 'cpu', 'cpu_opt']:
        file_path = f'profiling_results/{impl}_benchmark.txt'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                pattern = patterns.get(impl)
                if pattern:
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        if impl == 'cpu' or impl == 'cpu_opt':
                            accuracy, throughput = match.groups()
                        else:
                            throughput, accuracy = match.groups()
                        
                        results[impl] = {
                            'throughput': float(throughput),
                            'accuracy': float(accuracy)
                        }
                        print(f"Throughput={throughput}, Accuracy={accuracy}")
                    else:
                        print(f"pattern: {pattern}")
                else:
                    print(f"no {impl}")
    
    return results

def parse_time_profiles():
    results = {}
    
    for impl in ['cpu', 'cpu_opt']:
        file_path = f'profiling_results/{impl}_time_profile.txt'
        if os.path.exists(file_path):
            metrics = {}
            with open(file_path, 'r') as f:
                for line in f:
                    if 'User time' in line:
                        metrics['user_time'] = float(line.split(':')[1].strip())
                    elif 'System time' in line:
                        metrics['system_time'] = float(line.split(':')[1].strip())
                    elif 'Maximum resident set size' in line:
                        metrics['max_rss'] = int(line.split(':')[1].strip())
                    elif 'Elapsed (wall clock) time' in line:
                        time_str = line.split(':', 1)[1].strip()
                        nums = re.findall(r'\d+(?:\.\d+)?', time_str)
                        try:
                            if len(nums) == 3:
                                h, m, s = nums
                                metrics['wall_time'] = float(h)*3600 + float(m)*60 + float(s)
                            elif len(nums) == 2:
                                m, s = nums
                                metrics['wall_time'] = float(m)*60 + float(s)
                            elif len(nums) == 1:
                                metrics['wall_time'] = float(nums[0])
                            else:
                                print(f"Unrecognized time format: '{time_str}' for {impl}")
                                metrics['wall_time'] = 0.0
                        except ValueError as e:
                            print(f"Error parsing time '{time_str}' for {impl}: {e}")
                            metrics['wall_time'] = 0.0            
            if metrics:
                results[impl] = metrics
    
    return results

def analyze_nvprof_output():
    kernel_data = {}
    
    for filename in os.listdir('profiling_results'):
        if filename.startswith('nvprof_output_') and filename.endswith('.txt'):
            file_path = os.path.join('profiling_results', filename)
            
            with open(file_path, 'r') as f:
                content = f.read()
                
                # Extract kernel timing information
                kernel_section = False
                for line in content.split('\n'):
                    if "GPU activities:" in line:
                        kernel_section = True
                        continue
                    
                    if kernel_section and line.strip() and "%" in line:
                        parts = line.strip().split()
                        if len(parts) >= 7:
                            try:
                                # Extract kernel info
                                percentage = float(parts[0].strip('%'))
                                time_ms = float(parts[3])
                                kernel_name = parts[6]
                                
                                if kernel_name not in kernel_data:
                                    kernel_data[kernel_name] = []
                                kernel_data[kernel_name].append(time_ms)
                            except (ValueError, IndexError):
                                continue
    
    avg_kernel_times = {}
    for kernel, times in kernel_data.items():
        if times:
            avg_kernel_times[kernel] = sum(times) / len(times)
    
    return avg_kernel_times

def parse_memory_data():
    memory_data = {}
    
    for impl in ['gpu', 'gpu_nchw', 'cpu', 'cpu_opt']:
        file_path = f'profiling_results/{impl}_memory.csv'
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    memory_data[impl] = {
                        'peak_rss_mb': df['rss_kb'].max() / 1024,
                        'peak_vsz_mb': df['vsz_kb'].max() / 1024,
                        'avg_rss_mb': df['rss_kb'].mean() / 1024,
                        'timeline': {
                            'timestamps': df['timestamp'].values,
                            'rss_mb': df['rss_kb'].values / 1024
                        }
                    }
            except Exception as e:
                print(f"Error parsing memory data for {impl}: {e}")
    
    return memory_data

def create_visualizations(benchmark_results, time_results, kernel_times, memory_data):
    plt.style.use('ggplot')
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    implementations = list(benchmark_results.keys())
    throughputs = [benchmark_results[impl]['throughput'] for impl in implementations]
    
    bars = plt.bar(implementations, throughputs)
    plt.title('Throughput Comparison', fontsize=14)
    plt.ylabel('Images per second', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar, value in zip(bars, throughputs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(throughputs),
                 f'{value:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.subplot(2, 2, 2)
    accuracies = [benchmark_results[impl]['accuracy'] for impl in implementations]
    
    bars = plt.bar(implementations, accuracies)
    plt.title('Accuracy Comparison (All near ~96-97%)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=12)
    
    plt.ylim(0, 100)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar, value in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{value:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.subplot(2, 2, 3)
    if memory_data:
        impls = list(memory_data.keys())
        peak_memory = [memory_data[impl]['peak_rss_mb'] for impl in impls]
        avg_memory = [memory_data[impl]['avg_rss_mb'] for impl in impls]
        
        x = np.arange(len(impls))
        width = 0.35
        
        plt.bar(x - width/2, peak_memory, width, label='Peak Memory')
        plt.bar(x + width/2, avg_memory, width, label='Average Memory')
        plt.title('Memory Usage Comparison', fontsize=14)
        plt.xlabel('Implementation', fontsize=12)
        plt.ylabel('Memory (MB)', fontsize=12)
        plt.xticks(x, impls)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 4)
    if time_results:
        impls = list(time_results.keys())
        user_times = [time_results[impl]['user_time'] for impl in impls]
        system_times = [time_results[impl]['system_time'] for impl in impls]
        wall_times = [time_results[impl]['wall_time'] for impl in impls]
        
        x = np.arange(len(impls))
        width = 0.25
        
        plt.bar(x - width, user_times, width, label='User Time')
        plt.bar(x, system_times, width, label='System Time')
        plt.bar(x + width, wall_times, width, label='Wall Clock Time')
        plt.title('Execution Time Breakdown', fontsize=14)
        plt.xlabel('Implementation', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.xticks(x, impls)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('profiling_results/performance_comparison.png', dpi=300)
    
    if kernel_times:
        plt.figure(figsize=(12, 6))
        
        sorted_kernels = sorted(kernel_times.items(), key=lambda x: x[1], reverse=True)
        kernel_names = [name for name, _ in sorted_kernels]
        kernel_times_ms = [time for _, time in sorted_kernels]
        
        bars = plt.barh(kernel_names, kernel_times_ms)
        plt.title('CUDA Kernel Execution Times', fontsize=14)
        plt.xlabel('Time (ms)', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        for bar, time in zip(bars, kernel_times_ms):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{time:.3f} ms', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('profiling_results/gpu_kernel_analysis.png', dpi=300)
    
    if memory_data:
        plt.figure(figsize=(12, 6))
        
        for impl, data in memory_data.items():
            if 'timeline' in data:
                timeline = data['timeline']
                start_time = timeline['timestamps'][0]
                adjusted_times = [t - start_time for t in timeline['timestamps']]
                
                plt.plot(adjusted_times, timeline['rss_mb'], label=impl)
        
        plt.title('Memory Usage Timeline', fontsize=14)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Memory Usage (MB)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('profiling_results/memory_timeline.png', dpi=300)

    print("\nGenerated visualization charts in profiling_results/ directory")

def main():
    os.makedirs('profiling_results', exist_ok=True)
    
    print("Analyzing profiling results...")
    
    benchmark_results = parse_benchmark_files()
    print("Benchmark results:", benchmark_results)
    
    time_results = parse_time_profiles()
    print("Time profiling results:", time_results)
    
    kernel_times = analyze_nvprof_output()
    if kernel_times:
        print(f"Analyzed {len(kernel_times)} CUDA kernels")
    
    memory_data = parse_memory_data()
    if memory_data:
        print("Memory usage results:")
        for impl, data in memory_data.items():
            print(f"  {impl}: Peak: {data['peak_rss_mb']:.2f} MB, Avg: {data['avg_rss_mb']:.2f} MB")
    
    create_visualizations(benchmark_results, time_results, kernel_times, memory_data)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
EOF

chmod +x analyze_results.py

echo "analysis script"
python3 analyze_results.py

echo "Results saved in profiling_results/ directory."
