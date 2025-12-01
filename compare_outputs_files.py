"""
python compare_outputs_files.py --files output_bs1.pt output_bs2.pt output_bs4.pt
"""
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--files', nargs='+', required=True, help='Output files to compare')
args = parser.parse_args()

print(f"Comparing {len(args.files)} output files:")
for f in args.files:
    print(f"  - {f}")
print()

outputs = []
for f in args.files:
    output = torch.load(f)
    outputs.append(output)
    has_nan = torch.isnan(output).any().item()
    print(f"{f}:")
    print(f"  Shape: {output.shape}")
    print(f"  [0,:5]: {output[0, :5]}")
    print(f"  Has NaN: {has_nan}")
    print()

# Compare all outputs to the first one
print("Comparisons:")
for i in range(1, len(outputs)):
    match = torch.allclose(outputs[0], outputs[i])
    max_diff = (outputs[0] - outputs[i]).abs().max().item()
    print(f"  {args.files[0]} vs {args.files[i]}: {'✓ MATCH' if match else '✗ DIFFER'} (max_diff={max_diff})")
