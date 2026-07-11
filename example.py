"""TraceTorch example — trace a simple transformer model."""

import torch
import torch.nn as nn

from tracetorch import TraceSession


class SimpleTransformer(nn.Module):
    """A minimal transformer for demonstration."""

    def __init__(self, vocab_size: int = 1000, d_model: int = 768, n_heads: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.linear = nn.Linear(d_model, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x, _ = self.attention(x, x, x)
        x = x.mean(dim=1)
        x = self.linear(x)
        return x


def main() -> None:
    model = SimpleTransformer()
    session = TraceSession(model)

    with session:
        input_ids = torch.randint(0, 1000, (8, 512))
        model(input_ids)

    print(session.summary())
    print()

    session.export("./trace.json")
    print("Trace exported to ./trace.json")


if __name__ == "__main__":
    main()
