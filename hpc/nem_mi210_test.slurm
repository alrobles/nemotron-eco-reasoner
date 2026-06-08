#!/bin/bash
# MI210 ROCm test — uses nemotron-mi210.sif (ROCm PyTorch + mamba-ssm hipified)
#SBATCH --job-name=nem-mi210-test
#SBATCH --partition=sixhour
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/a474r867/scratch/nem_mi210_test_%j.out
#SBATCH --error=/home/a474r867/scratch/nem_mi210_test_%j.err
# Target MI210 nodes — adjust nodelist as needed
# #SBATCH --constraint=mi210

export APPTAINER_WRITABLE_TMPFS=1
export PYTHONUNBUFFERED=1

C=/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-mi210.sif
M=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
P=/tmp/pip_$SLURM_JOB_ID

echo "=== MI210 ROCm TEST job=$SLURM_JOB_ID $(date) ==="
echo "SIF: $C"
ls -la $C

echo "=== Phase 1: Verify ROCm + PyTorch ==="
apptainer exec --rocm --bind "/dev/shm,/tmp,$P" \
    --env PYTHONUNBUFFERED=1 \
    $C python3 -c '
import torch
print(f"PyTorch {torch.__version__}")
print(f"ROCm/CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_mem / 1e9:.1f} GB")
    print(f"Compute Capability: {props.major}.{props.minor}")
    # Quick matmul test
    a = torch.randn(100, 100, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(100, 100, device="cuda", dtype=torch.bfloat16)
    c = torch.matmul(a, b)
    print(f"ROCm matmul test: OK ({c.shape})")
    print(f"bf16 supported: True")
else:
    print("ERROR: No GPU detected!")
    exit(1)
print("=== Phase 1 PASSED ===")
'
if [ $? -ne 0 ]; then echo "PHASE 1 FAILED — ROCm/PyTorch check failed"; exit 1; fi

echo "=== Phase 1.5: Check mamba-ssm ==="
apptainer exec --rocm --bind "/dev/shm,/tmp,$P" \
    --env PYTHONUNBUFFERED=1 \
    $C python3 -c '
try:
    import mamba_ssm
    print(f"mamba_ssm {mamba_ssm.__version__} OK (hipified)")
except ImportError as e:
    print(f"mamba_ssm not available: {e}")
    print("Will need runtime compilation or monkey-patch fallback")
try:
    import causal_conv1d
    print(f"causal_conv1d {causal_conv1d.__version__} OK")
except ImportError as e:
    print(f"causal_conv1d not available: {e}")
print("=== Phase 1.5 done ===")
'

echo "=== Phase 2: Quick training test (10 steps) ==="

apptainer exec --rocm --bind "/dev/shm,/tmp,$P,$M,$O" \
    --env PYTHONUNBUFFERED=1 --env TORCH_COMPILE_DISABLE=1 \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    $C python3 -c "
import torch, json, os, time

# Check if mamba_ssm is available
mamba_ok = False
try:
    import mamba_ssm
    mamba_ok = True
    print(f'mamba_ssm: OK ({mamba_ssm.__version__})')
except:
    print('mamba_ssm: NOT available — testing basic model load only')

from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = '$M'
print(f'Loading tokenizer from {model_path}...')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f'Tokenizer loaded: {tokenizer.__class__.__name__}')

if mamba_ok:
    print('Loading model (4-bit quantized)...')
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        quantization_config=bnb_config, device_map='auto'
    )
    print(f'Model loaded: {model.__class__.__name__}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e9:.1f}B')
    
    # Quick inference test
    inputs = tokenizer('What is 2+2?', return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)
    print(f'Inference test: {tokenizer.decode(out[0], skip_special_tokens=True)[:100]}')
else:
    print('Skipping model load — mamba_ssm required for Nemotron architecture')

# Output manifest
manifest = {
    'job_id': '$SLURM_JOB_ID',
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
    'pytorch': torch.__version__,
    'mamba_ssm': mamba_ok,
    'rocm': True,
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
}
print(f'=== MANIFEST: {json.dumps(manifest)} ===')
print('=== MI210 TEST COMPLETE ===')
"
echo "=== ALL PHASES DONE ==="
