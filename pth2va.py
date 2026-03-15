"""
pth2va.py -- Convert PyTorch .pth weights to Verilog-A ANN forward-pass code snippet.

Automatically detects network architecture (layer sizes, activation functions)
from the state_dict keys. Supports nn.Sequential and named-layer models.

Supported activations: tanh, LeakyReLU, ReLU (auto-detected or user-specified).

Input:  One or two .pth files + optional va_scalers.json
Output: Verilog-A code snippet with explicit algebraic ANN forward pass.

Usage:
    # Two directions with scalers
    python pth2va.py --dir0 dir0.pth --dir1 dir1.pth --scalers va_scalers.json

    # Single model, specify activation and input names
    python pth2va.py --model my_model.pth --act tanh --inputs "V,FE" --out snippet.va

    # Auto-detect everything
    python pth2va.py --dir0 dir0.pth --dir1 dir1.pth
"""
import argparse
import json
import re
import torch


# -- Parse state_dict to extract linear layers --

def parse_layers(state):
    """
    Auto-detect linear layers from state_dict keys.
    Returns list of (weight_2d, bias_1d) in forward order.
    Handles key patterns like:
      - fc1.weight / fc1.bias
      - net.0.weight / net.0.bias
      - layers.0.weight
      - 0.weight
    """
    # Find all weight keys
    weight_keys = sorted([k for k in state if k.endswith('.weight')],
                         key=lambda k: _layer_order(k))
    layers = []
    for wk in weight_keys:
        bk = wk.replace('.weight', '.bias')
        W = state[wk]
        if W.dim() != 2:
            continue  # skip non-linear layers (e.g. BatchNorm)
        b = state.get(bk)
        if b is None or b.dim() != 1:
            continue
        layers.append((_to_list2d(W), _to_list1d(b)))
    return layers


def _layer_order(key):
    """Extract numeric order from key for sorting."""
    nums = re.findall(r'(\d+)', key)
    return [int(n) for n in nums] if nums else [ord(c) for c in key]


def _to_list2d(t):
    return [[float(t[i][j]) for j in range(t.shape[1])] for i in range(t.shape[0])]


def _to_list1d(t):
    return [float(v) for v in t]


# -- Activation functions in Verilog-A --

ACT_VA = {
    'tanh':      lambda var: f"tanh({var})",
    'relu':      lambda var: f"(({var}) > 0.0 ? ({var}) : 0.0)",
    'leakyrelu': lambda var: f"(({var}) > 0.0 ? ({var}) : 0.01 * ({var}))",
}


def detect_activation(state):
    """Guess activation from state_dict keys (heuristic)."""
    keys_str = " ".join(state.keys()).lower()
    # If keys contain hints like "leaky" or the model has dropout (common with LeakyReLU)
    # Default guess based on common patterns
    for hint, act in [('leaky', 'leakyrelu'), ('relu', 'relu'), ('tanh', 'tanh')]:
        if hint in keys_str:
            return act
    return 'tanh'  # safe default


# -- Code generation --

def gen_snippet(layers, tag, act_name, scalers=None, input_names=None):
    """
    Generate Verilog-A forward-pass for one set of layers.

    layers: list of (W, b) tuples
    tag: suffix like 'd0', 'd1', or ''
    act_name: 'tanh', 'relu', 'leakyrelu'
    scalers: dict with 'xmin', 'xrng', 'ymin', 'yrng' (optional)
    input_names: list of input variable names (optional)
    """
    act_fn = ACT_VA[act_name]
    n_layers = len(layers)
    n_in = len(layers[0][0][0])  # first layer input size
    n_out = len(layers[-1][0])   # last layer output size
    suffix = f"_{tag}" if tag else ""

    lines = []
    a = lines.append

    sizes = [n_in] + [len(l[0]) for l in layers]
    a(f"    // -- Forward pass [{tag}]: {'->'.join(str(s) for s in sizes)}, act={act_name} --")

    # Determine input expressions
    if scalers and input_names:
        xmin = scalers['xmin']
        xrng = scalers['xrng']
        inp_exprs = [f"(({nm}) - ({xmin[i]:.10e})) / ({xrng[i]:.10e})"
                     for i, nm in enumerate(input_names)]
    elif input_names:
        inp_exprs = list(input_names)
    else:
        inp_exprs = [f"x{suffix}[{i}]" for i in range(n_in)]

    # Generate layer by layer
    for L, (W, b) in enumerate(layers):
        n_out_l = len(W)
        n_in_l = len(W[0])
        is_last = (L == n_layers - 1)
        lname = f"h{L+1}{suffix}" if not is_last else f"out{suffix}"

        for j in range(n_out_l):
            # Build weighted sum
            if L == 0:
                terms = " + ".join(f"{W[j][i]:.10e} * {inp_exprs[i]}" for i in range(n_in_l))
            else:
                prev = f"h{L}{suffix}"
                terms = " + ".join(f"{W[j][i]:.10e} * {prev}[{i}]" for i in range(n_in_l))
            expr = f"{terms} + ({b[j]:.10e})"

            if is_last:
                # Output layer: no activation, apply output de-scaling
                if scalers:
                    a(f"    {lname} = ({expr}) * {scalers['yrng']:.10e} + ({scalers['ymin']:.10e});")
                else:
                    a(f"    {lname} = {expr};")
            else:
                a(f"    {lname}[{j}] = {act_fn(expr)};")

    a("")
    return "\n".join(lines)


def gen_declarations(layers, tag):
    """Generate Verilog-A variable declarations for one network."""
    suffix = f"_{tag}" if tag else ""
    n_layers = len(layers)
    decls = []
    for L in range(n_layers - 1):  # hidden layers only
        size = len(layers[L][0])
        decls.append(f"real h{L+1}{suffix}[0:{size-1}];")
    decls.append(f"real out{suffix};")
    return decls


# -- Main --

def main():
    parser = argparse.ArgumentParser(
        description='Auto-convert PyTorch .pth to Verilog-A ANN snippet')
    parser.add_argument('--model', help='Single model .pth file')
    parser.add_argument('--dir0', help='Direction 0 model .pth')
    parser.add_argument('--dir1', help='Direction 1 model .pth')
    parser.add_argument('--scalers', help='va_scalers.json (optional)')
    parser.add_argument('--act', choices=['tanh', 'relu', 'leakyrelu'],
                        help='Activation function (auto-detected if omitted)')
    parser.add_argument('--inputs', help='Comma-separated input variable names, e.g. "restored_q,FE_nm"')
    parser.add_argument('--out', default='ann_weights.va', help='Output file')
    args = parser.parse_args()

    # Load scalers
    scalers = None
    if args.scalers:
        with open(args.scalers) as f:
            scalers = json.load(f)

    input_names = args.inputs.split(',') if args.inputs else None

    # Determine mode: single model or two directions
    pth_files = {}
    if args.model:
        pth_files[''] = args.model
    else:
        if args.dir0:
            pth_files['d0'] = args.dir0
        if args.dir1:
            pth_files['d1'] = args.dir1

    if not pth_files:
        parser.error('Provide --model or --dir0/--dir1')

    all_lines = []
    all_decls = []

    for tag, path in pth_files.items():
        state = torch.load(path, map_location='cpu', weights_only=True)
        layers = parse_layers(state)

        if not layers:
            print(f"ERROR: No linear layers found in {path}")
            print(f"  Keys: {list(state.keys())}")
            return

        # Detect activation
        act = args.act or detect_activation(state)

        # Architecture summary
        sizes = [len(layers[0][0][0])] + [len(l[0]) for l in layers]
        print(f"  {path}: {'->'.join(str(s) for s in sizes)}, act={act}")

        # Get per-direction scalers
        sc = None
        if scalers:
            # Try keyed by direction index
            d_idx = tag.replace('d', '') if tag else '0'
            sc = scalers.get(d_idx, scalers.get(tag, None))

        all_decls.extend(gen_declarations(layers, tag))
        all_lines.append(gen_snippet(layers, tag, act, sc, input_names))

    # Write output
    with open(args.out, 'w') as f:
        f.write("// Auto-generated Verilog-A ANN forward pass\n")
        f.write("// Variable declarations:\n")
        for d in all_decls:
            f.write(f"//   {d}\n")
        f.write("\n")
        for block in all_lines:
            f.write(block)
            f.write("\n")

    print(f"=> {args.out}")


if __name__ == '__main__':
    main()
