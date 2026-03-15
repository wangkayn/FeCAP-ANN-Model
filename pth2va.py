"""
pth2va.py -- Convert PyTorch .pth weights to Verilog-A ANN forward-pass code snippet.

Input:  Two .pth files (direction 0 and 1) + va_scalers.json
Output: Pure ANN weight-unrolled Verilog-A snippet (h1/h2/out only, no module wrapper)

Usage:
    python pth2va.py
    python pth2va.py --dir0 best_va_dir0.pth --dir1 best_va_dir1.pth --scalers va_scalers.json --out ann_weights.va
"""
import argparse
import json
import torch
import torch.nn as nn

NH = 16
NI = 2


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NI, NH)
        self.fc2 = nn.Linear(NH, NH)
        self.fc3 = nn.Linear(NH, 1)
    def forward(self, x):
        return self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(x)))))


def load_weights(pth_path):
    state = torch.load(pth_path, map_location='cpu')
    def to2d(t): return [[float(t[i][j]) for j in range(t.shape[1])] for i in range(t.shape[0])]
    def to1d(t): return [float(v) for v in t]
    return dict(
        W1=to2d(state['fc1.weight']), b1=to1d(state['fc1.bias']),
        W2=to2d(state['fc2.weight']), b2=to1d(state['fc2.bias']),
        W3=to2d(state['fc3.weight']), b3=to1d(state['fc3.bias']),
    )


def gen_forward_pass(w, scalers):
    """Generate ANN forward-pass code snippet only."""
    lines = []
    a = lines.append

    for d, tag in [(0, 'd0'), (1, 'd1')]:
        sc = scalers[str(d)]
        W1, b1 = w[d]['W1'], w[d]['b1']
        W2, b2 = w[d]['W2'], w[d]['b2']
        W3, b3 = w[d]['W3'], w[d]['b3']

        inp0 = f"(restored_q - ({sc['xmin'][0]:.10e})) / ({sc['xrng'][0]:.10e})"
        inp1 = f"(FE_nm - ({sc['xmin'][1]:.10e})) / ({sc['xrng'][1]:.10e})"

        a(f"    // -- Direction {d} forward pass --")
        for j in range(NH):
            a(f"    h1_{tag}[{j}] = tanh({W1[j][0]:.10e} * {inp0} + {W1[j][1]:.10e} * {inp1} + ({b1[j]:.10e}));")
        for j in range(NH):
            expr = " + ".join(f"{W2[j][i]:.10e} * h1_{tag}[{i}]" for i in range(NH))
            a(f"    h2_{tag}[{j}] = tanh({expr} + ({b2[j]:.10e}));")
        expr = " + ".join(f"{W3[0][i]:.10e} * h2_{tag}[{i}]" for i in range(NH))
        a(f"    out_{tag} = ({expr} + ({b3[0]:.10e})) * {sc['yrng']:.10e} + ({sc['ymin']:.10e});")
        a("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Convert .pth weights to Verilog-A ANN snippet')
    parser.add_argument('--dir0', default='best_va_dir0.pth')
    parser.add_argument('--dir1', default='best_va_dir1.pth')
    parser.add_argument('--scalers', default='va_scalers.json')
    parser.add_argument('--out', default='ann_weights.va')
    args = parser.parse_args()

    w0 = load_weights(args.dir0)
    w1 = load_weights(args.dir1)
    with open(args.scalers) as f:
        scalers = json.load(f)

    snippet = gen_forward_pass({0: w0, 1: w1}, scalers)

    with open(args.out, 'w') as f:
        f.write(snippet)
    print(f"=> {args.out}")


if __name__ == '__main__':
    main()
