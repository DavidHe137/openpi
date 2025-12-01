import torch
from pi0_infer import Pi0Inference
import numpy as np

# Use same setup as benchmark
np.random.seed(100)
torch.manual_seed(100)

# Create inputs once and reuse for all batch sizes
single_image = torch.randn(1, 2, 224, 224, 3, dtype=torch.bfloat16).cuda()
single_state = torch.randn(1, 32, dtype=torch.bfloat16).cuda()
single_noise = torch.randn(1, 63, 32, dtype=torch.bfloat16).cuda()

for bs in [4]:
    print(f"\n{'=' * 60}")
    print(f"Testing batch_size={bs}")
    print("=" * 60)

    infer = Pi0Inference(
        {
            "language_embeds": torch.randn(0, 2048, dtype=torch.bfloat16),
        },
        num_views=2,
        chunk_size=63,
        batch_size=bs,
    )

    # Repeat the same inputs across batch dimension
    input_image = single_image.repeat(bs, 1, 1, 1, 1)
    input_state = single_state.repeat(bs, 1)
    input_noise = single_noise.repeat(bs, 1, 1)

    output = infer.forward(input_image, input_state, input_noise)
    torch.cuda.synchronize()

    has_nan = torch.isnan(output).any().item()
    print(f"Output shape: {output.shape}")
    print(f"Output[0,0,:5]: {output[0, 0, :5]}")
    print(f"Has NaN: {has_nan}")

    if has_nan:
        print("✗ CONTAINS NaN!")
    else:
        print("✓ Valid output")
