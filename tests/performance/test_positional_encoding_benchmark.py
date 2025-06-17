"""
Compact benchmark test for RotaryEncoder vs ReferenceRotaryEncoder implementations.
"""

import time

import pytest
import torch

from fairseq2.nn import ReferenceRotaryEncoder, RotaryEncoder
from fairseq2.nn._batch_layout import BatchLayout


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    else: return torch.device("cpu")

def sync_device(device):
    if device.type == "cuda": torch.cuda.synchronize()
    elif device.type == "mps": torch.mps.synchronize()

def create_test_batch(batch_size: int, seq_len: int, encoding_dim: int, device: torch.device):
    seqs = torch.randn(batch_size, seq_len, encoding_dim, dtype=torch.float32, device=device)
    batch_layout = BatchLayout((batch_size, seq_len), [seq_len] * batch_size, device=device)
    return seqs, batch_layout

def benchmark_encoder(encoder, seqs, batch_layout, device, num_runs=50):
    # Warmup
    for _ in range(5):
        sync_device(device)
        encoder(seqs, batch_layout)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        sync_device(device)
        start = time.perf_counter()
        encoder(seqs, batch_layout)
        sync_device(device)
        times.append(time.perf_counter() - start)
    
    return sum(times) / len(times)

benchmark_results = []

@pytest.mark.parametrize("encoding_dim", [1024, 2048, 4096])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [4096, 16384])
@pytest.mark.parametrize("compile", [False, True])
def test_benchmark_positional_encoders(encoding_dim: int, batch_size: int, seq_len: int, compile: bool):
    device = get_device()
    max_seq_len = 32768
    theta = 1_000_000.0
    
    # Create encoders
    rotary_encoder = RotaryEncoder(encoding_dim, max_seq_len, theta=theta, device=device)
    reference_encoder = ReferenceRotaryEncoder(encoding_dim, max_seq_len, theta=theta, device=device)
    
    # Create test data
    seqs, batch_layout = create_test_batch(batch_size, seq_len, encoding_dim, device)
    
    if compile:
        rotary_encoder = torch.compile(rotary_encoder, backend="inductor", dynamic=True)
        reference_encoder = torch.compile(reference_encoder, backend="inductor", dynamic=True)

    # Benchmark
    rotary_time = benchmark_encoder(rotary_encoder, seqs, batch_layout, device)
    reference_time = benchmark_encoder(reference_encoder, seqs, batch_layout, device)
    
    speedup = reference_time / rotary_time
    print(f"B={batch_size} S={seq_len} E={encoding_dim} | Rotary: {rotary_time*1000:.1f}ms | Ref: {reference_time*1000:.1f}ms | {speedup:.2f}x")
    
    benchmark_results.append({
        'batch_size': batch_size, 'seq_len': seq_len, 'encoding_dim': encoding_dim,
        'rotary_time': rotary_time, 'reference_time': reference_time, 'speedup': speedup
    })

def test_torch_compile_comparison():
    device = get_device()
    if device.type != "cuda":
        print(f"Skipping compile test - need CUDA, got {device.type}")
        return
    
    # Test config: realistic large-scale settings
    encoding_dim = 2048
    max_seq_len = 32768
    batch_size = 2
    seq_len = 16384
    theta = 1_000_000.0
    
    seqs, batch_layout = create_test_batch(batch_size, seq_len, encoding_dim, device)
    
    print(f"\nCompile Test: B={batch_size} S={seq_len} E={encoding_dim} theta={theta}")
    
    for name, encoder_class in [("Rotary", RotaryEncoder), ("Reference", ReferenceRotaryEncoder)]:
        encoder = encoder_class(encoding_dim, max_seq_len, theta=theta, device=device)
        
        # Eager backend (baseline)
        eager_encoder = torch.compile(encoder, backend="eager", dynamic=True)
        eager_time = benchmark_encoder(eager_encoder, seqs, batch_layout, device, num_runs=20)
        
        # Inductor backend
        compiled_encoder = torch.compile(encoder, backend="inductor", dynamic=True)
        inductor_time = benchmark_encoder(compiled_encoder, seqs, batch_layout, device, num_runs=20)
        
        compile_speedup = eager_time / inductor_time
        print(f"{name:9} | Eager: {eager_time*1000:.1f}ms | Inductor: {inductor_time*1000:.1f}ms | {compile_speedup:.2f}x")

def test_numerical_accuracy(compile_encoders=False):
    """Test numerical accuracy between implementations across sequence lengths."""
    device = get_device()
    encoding_dim = 256 
    max_seq_len = 1_000_000  # 1M tokens
    theta = 1_000_000.0
    
    mode_str = "Compiled" if compile_encoders else "Eager"
    print(f"\nNumerical Accuracy Test - {mode_str} Mode (device: {device.type})")
    print(f"Encoding dim: {encoding_dim}, Max seq len: {max_seq_len:,}")
    
    # Create encoders
    rotary_encoder = RotaryEncoder(encoding_dim, max_seq_len, theta=theta, device=device)
    reference_encoder = ReferenceRotaryEncoder(encoding_dim, max_seq_len, theta=theta, device=device)
    
    # Compile if requested
    if compile_encoders:
        if device.type == "cuda":
            rotary_encoder = torch.compile(rotary_encoder, backend="inductor", dynamic=True)
            reference_encoder = torch.compile(reference_encoder, backend="inductor", dynamic=True)
        else:
            print(f"Skipping compilation - requires CUDA, got {device.type}")
            return
    
    # Test different sequence lengths
    seq_lengths = [1024, 4096, 16384, 65536, 262144, 1_000_000]  # Up to 1M
    batch_size = 1  # Keep small for memory
    
    print(f"\n{'SeqLen':>8} | {'MaxDiff':>10} | {'MeanDiff':>10} | {'StdDiff':>10} | {'Status':>8}")
    print("-" * 55)
    
    for seq_len in seq_lengths:
        if seq_len > max_seq_len:
            continue
            
        try:
            # Create test data
            seqs, batch_layout = create_test_batch(batch_size, seq_len, encoding_dim, device)
            
            # Get outputs from both encoders
            with torch.no_grad():
                rotary_output = rotary_encoder(seqs, batch_layout)
                reference_output = reference_encoder(seqs, batch_layout)
            
            # Calculate differences
            diff = torch.abs(rotary_output - reference_output)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            std_diff = torch.std(diff).item()
            
            # Determine status
            if max_diff < 1e-6:
                status = "PERFECT"
            elif max_diff < 1e-5:
                status = "GOOD"
            elif max_diff < 1e-4:
                status = "OK"
            else:
                status = "WARN"
            
            print(f"{seq_len:>8,} | {max_diff:>9.2e} | {mean_diff:>9.2e} | {std_diff:>9.2e} | {status:>8}")
            
            # Clean up memory for large sequences
            del seqs, batch_layout, rotary_output, reference_output, diff
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{seq_len:>8,} | {'OOM':>9} | {'OOM':>9} | {'OOM':>9} | {'SKIP':>8}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                print(f"{seq_len:>8,} | {'ERROR':>9} | {'ERROR':>9} | {'ERROR':>9} | {'FAIL':>8}")
        except Exception as e:
            print(f"{seq_len:>8,} | {'ERROR':>9} | {'ERROR':>9} | {'ERROR':>9} | {'FAIL':>8}")


def test_numerical_accuracy_compiled():
    """Test numerical accuracy with torch.compile enabled."""
    test_numerical_accuracy(compile_encoders=True)


def test_zzz_benchmark_summary():
    if not benchmark_results:
        return
    
    print(f"\nSummary ({len(benchmark_results)} configs):")
    speedups = [r['speedup'] for r in benchmark_results]
    print(f"Speedup: avg={sum(speedups)/len(speedups):.2f}x min={min(speedups):.2f}x max={max(speedups):.2f}x")

if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")
    
    # Quick test
    configs = [(2048, 2, 8192), (4096, 4, 16384)]
    for encoding_dim, batch_size, seq_len in configs:
        test_benchmark_positional_encoders(encoding_dim, batch_size, seq_len)
    
    test_torch_compile_comparison()
    test_numerical_accuracy()
    test_numerical_accuracy_compiled()
    test_zzz_benchmark_summary()
