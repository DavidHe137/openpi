import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class UNet(nn.Module):
    """Simple U-Net architecture for vector field prediction"""

    def __init__(self, input_channels=4):  # 3 for image + 1 for time
        super().__init__()

        # Encoder
        self.enc1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.enc3 = nn.Conv2d(128, 256, 3, padding=1, stride=2)

        # Decoder
        self.dec3 = nn.ConvTranspose2d(256, 128, 4, padding=1, stride=2)
        self.dec2 = nn.ConvTranspose2d(256, 64, 4, padding=1, stride=2)
        self.dec1 = nn.Conv2d(128, 3, 3, padding=1)  # Output vector field

    def forward(self, x, t):
        # Append time channel
        t = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        x = torch.cat([x, t], dim=1)

        # Encoder
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))

        # Decoder with skip connections
        d3 = F.relu(self.dec3(e3))
        d2 = F.relu(self.dec2(torch.cat([d3, e2], dim=1)))
        return self.dec1(torch.cat([d2, e1], dim=1))


class FlowMatching(nn.Module):
    def __init__(self):
        super().__init__()
        self.vector_field = UNet()

    def get_flow_field(self, x0, x1, t):
        """Compute flow field between x0 and x1 at time t"""
        xt = x0 + t * (x1 - x0)  # Linear interpolation
        return self.vector_field(xt, t)

    def loss_function(self, x0, x1):
        """Compute flow matching loss"""
        batch_size = x0.shape[0]
        t = torch.rand(batch_size, device=x0.device)  # Random time steps

        # Get predicted vector field
        v_pred = self.get_flow_field(x0, x1, t.view(-1, 1, 1, 1))

        # Ground truth vector field (x1 - x0 for linear paths)
        v_true = x1 - x0

        # L2 loss between predicted and true vector fields
        return F.mse_loss(v_pred, v_true)

    @torch.no_grad()
    def sample(self, x0, steps=100):
        """Generate samples using Euler method"""
        dt = 1.0 / steps
        xt = x0

        for i in range(steps):
            t = torch.ones(x0.shape[0], device=x0.device) * i * dt
            v = self.get_flow_field(x0, None, t)
            xt = xt + v * dt

        return xt


def train_step(model, optimizer, x0, x1):
    """Single training step"""
    optimizer.zero_grad()
    loss = model.loss_function(x0, x1)
    loss.backward()
    optimizer.step()
    return loss.item()


# Example usage
def main():
    # Initialize model
    model = FlowMatching()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Example data (batch of 32 images of size 32x32)
    batch_size, channels, height, width = 32, 3, 32, 32
    x0 = torch.randn(batch_size, channels, height, width)  # Random noise
    x1 = torch.randn(batch_size, channels, height, width)  # Target images

    # Training loop
    n_epochs = 100
    for epoch in range(n_epochs):
        loss = train_step(model, optimizer, x0, x1)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Generate samples
    with torch.no_grad():
        samples = model.sample(x0)
        print("Generated samples shape:", samples.shape)


if __name__ == "__main__":
    main()
