# CLI Tools

tinygs provides several command-line executables for training, testing, and exporting configurations.

## config_train

The main training executable that builds and runs a training pipeline from a JSON configuration file.

### Usage

```bash
./config_train -c <config_file> [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-c, --config` | Path to JSON configuration file | (required) |
| `-v, --visualize` | Enable real-time visualization | `false` |
| `-d, --debug` | Enable debug logging | `false` |
| `-l, --log_level` | Log level: `debug`, `info`, `warn`, `error` | `warn` |
| `-h, --help` | Print help message | - |

### Examples

```bash
# Train with default settings
./config_train -c configs/garden.json

# Train with visualization
./config_train -c configs/garden.json -v

# Train with debug logging
./config_train -c configs/garden.json -d

# Train with info-level logging
./config_train -c configs/garden.json -l info
```

### Output

During training, progress is displayed:

```
STEP 100] PSNR=18.234 | TPUT=1234ms/100step | LR=1.0e+0 | N-Gs:   15234 | T= 12s
```

After training, evaluation results are saved to `out_dir/stats.json`:

```json
{
  "psnr": 25.67,
  "time": 240
}
```

### Workflow

The `config_train` executable:

1. Loads JSON configuration file
2. Creates and configures all components:
   - Dataset and DataLoader
   - Initialization from point cloud
   - Rasterizer
   - Optimizer and LR Scheduler
   - Loss functions and metrics
   - Strategy
   - Pose optimizer
3. Initializes Gaussians from input point cloud
4. Runs training loop via Orchestrator
5. Evaluates on test set
6. Saves results

---

## single_gs

A debugging/testing tool that renders a scene with one or two Gaussians. Useful for verifying rasterizer correctness.

### Usage

```bash
./single_gs [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-r, --rasterizer` | Rasterizer type | `default` |
| `--opacity1` | Opacity of first Gaussian | `0.6` |
| `--opacity2` | Opacity of second Gaussian | `0.6` |
| `--scale1` | Scale of first Gaussian | `0.1` |
| `--scale2` | Scale of second Gaussian | `0.1` |
| `-h, --help` | Print help message | - |

### Example

```bash
# Test default rasterizer
./single_gs -r default

# Test FastGS with custom opacity
./single_gs -r fastgs --opacity1 0.8 --opacity2 0.4
```

### Output

The tool prints:
- Color statistics (min/max/avg/std per channel)
- Gradient values for each Gaussian parameter
- Densification information

---

## export_default

Exports a default JSON configuration template.

### Usage

```bash
./export_default [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output file path | (stdout) |
| `-h, --help` | Print help message | - |

### Example

```bash
# Print to stdout
./export_default

# Save to file
./export_default -o my_config.json
```

### Output

Generates a complete JSON configuration with default values for all components:

```json
{
  "dataset": { ... },
  "dataloader": { ... },
  "initializer": { ... },
  "rasterizer": { ... },
  "optimizer": { ... },
  "lr_scheduler": { ... },
  "strategy": { ... },
  "losses": [ ... ],
  "metrics": [ ... ],
  "trainer": { ... }
}
```

---

## video_to_png

Converts video files to PNG image sequences for dataset preparation.

### Usage

```bash
./video_to_png <video_file> <output_folder>
```

---

## Building

All CLI tools are built using the build script:

```bash
# Build default targets
./build.sh

# Build with specific configuration
BUILD_TYPE=Debug ./build.sh

# Build specific target
TARGETS="config_train,single_gs" ./build.sh
```

The built executables are placed in the `build/` directory.