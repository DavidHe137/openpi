'''
python save_output.py --batch_size 4 --output_file output_bs4.pt
'''
import torch
from openpi.shared.pi0_infer import Pi0Inference
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--output_file', type=str, required=True)
args = parser.parse_args()

# Use same setup as benchmark
np.random.seed(100)
torch.manual_seed(100)

# Create inputs
single_image = torch.randn(1, 2, 224, 224, 3, dtype=torch.bfloat16).cuda()
single_state = torch.randn(1, 32, dtype=torch.bfloat16).cuda()
single_noise = torch.randn(1, 63, 32, dtype=torch.bfloat16).cuda()

print(f"Creating Pi0Inference with batch_size={args.batch_size}")
infer = Pi0Inference(
    {
        "language_embeds": torch.randn(0, 2048, dtype=torch.bfloat16),
    },
    num_views=2,
    chunk_size=63,
    batch_size=args.batch_size,
)

# Repeat inputs across batch dimension
input_image = single_image.repeat(args.batch_size, 1, 1, 1, 1)
input_state = single_state.repeat(args.batch_size, 1)
input_noise = single_noise.repeat(args.batch_size, 1, 1)

output = infer.forward(input_image, input_state, input_noise)
torch.cuda.synchronize()

has_nan = torch.isnan(output).any().item()
print(f"Output shape: {output.shape}")
print(f"Output[0,0,:5]: {output[0, 0, :5]}")
print(f"Has NaN: {has_nan}")

# Save first element to file
torch.save(output[0].cpu(), args.output_file)
print(f"Saved output[0] to {args.output_file}")
