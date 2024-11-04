import torch


class GEGLU(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim * 8)

    def forward(self, x):
        x = self.proj(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class Attention(torch.nn.Module):

    def __init__(self, heads, dim_q, dim_kv):
        super().__init__()

        self.heads = heads
        self.q = torch.nn.Linear(dim_q, 64 * heads, bias=False)
        self.k = torch.nn.Linear(dim_kv, 64 * heads, bias=False)
        self.v = torch.nn.Linear(dim_kv, 64 * heads, bias=False)
        self.out = torch.nn.Linear(64 * heads, dim_q)

    def forward(self, q, kv):
        q = self.q(q)
        k = self.k(kv)
        v = self.v(kv)

        #[b, dim, 64 * heads] -> [b * heads, dim, 64]
        q = q.unflatten(2, (self.heads, -1)).transpose(1, 2).flatten(end_dim=1)
        k = k.unflatten(2, (self.heads, -1)).transpose(1, 2).flatten(end_dim=1)
        v = v.unflatten(2, (self.heads, -1)).transpose(1, 2).flatten(end_dim=1)

        dtype = q.dtype
        atten = torch.empty(q.shape[0],
                            q.shape[1],
                            k.shape[1],
                            dtype=q.dtype,
                            device=q.device)

        atten = torch.baddbmm(atten,
                              q,
                              k.transpose(-1, -2),
                              beta=0,
                              alpha=64**-0.5)

        atten = atten.float().softmax(dim=-1).to(dtype)

        atten = torch.bmm(atten, v)

        #[b * heads, dim, 64] -> [b, dim, 64 * heads]
        atten = atten.unflatten(0, (-1, self.heads)).transpose(
            1, 2).flatten(start_dim=2)

        return self.out(atten)


class Transformer2D(torch.nn.Module):

    def __init__(self, heads, dim):
        super().__init__()

        self.norm1 = torch.nn.GroupNorm(32, dim, 1e-6, True)
        self.norm2 = torch.nn.LayerNorm(heads * 64, 1e-5, True)
        self.norm3 = torch.nn.LayerNorm(heads * 64, 1e-5, True)
        self.norm4 = torch.nn.LayerNorm(heads * 64, 1e-5, True)

        self.fc_in = torch.nn.Linear(dim, heads * 64)
        self.fc_out = torch.nn.Linear(heads * 64, dim)

        self.atten1 = Attention(heads=heads,
                                dim_q=heads * 64,
                                dim_kv=heads * 64)
        self.atten2 = Attention(heads=heads, dim_q=heads * 64, dim_kv=1024)

        self.ff = torch.nn.Sequential(
            GEGLU(heads * 64), torch.nn.Linear(heads * 64 * 4, heads * 64))

    def forward(self, q, kv):
        res = q
        size = q.shape[2:]

        q = self.norm1(q)

        q = q.flatten(start_dim=2).transpose(1, 2)

        q = self.fc_in(q)

        t = self.norm2(q)
        q = self.atten1(t, t) + q

        if kv is not None:
            q = self.atten2(self.norm3(q), kv) + q

        q = self.ff(self.norm4(q)) + q

        q = self.fc_out(q)

        q = q.transpose(1, 2).unflatten(2, size).contiguous()

        return q + res