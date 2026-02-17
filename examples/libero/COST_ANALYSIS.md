# Cost Analysis for Robot Scheduling

This module provides tools for analyzing the cost of robot action scheduling, enabling optimization of chunk request timing to minimize overall cost.

## Overview

The cost model consists of two components:

1. **Execution Cost**: Linear cost that accumulates as an action chunk is executed
   - Starts at 0 for each action chunk
   - Scales linearly with time: `cost = execution_cost_rate × duration`
   - Accumulates until the chunk ends or gets replanned

2. **Pause Cost**: One-time penalty incurred when there's a gap between action chunks
   - Triggered when the gap between chunks exceeds a threshold
   - Fixed cost added to the total
   - Common with SyncBroker due to synchronization delays

## Configuration Parameters

- `execution_cost_rate` (default: 1.0): Linear scaling factor for execution cost
- `pause_cost` (default: 10.0): Fixed cost added for each pause
- `pause_threshold` (default: 0.05): Gap duration (seconds) to consider as a pause

## Usage

### Real-time Cost Analysis (During Execution)

Enable cost analysis when running experiments by adding the `--enable-cost-analysis` flag:

```bash
python examples/libero/main_multi_robot_runtime.py \
    --enable-cost-analysis \
    --execution-cost-rate 1.0 \
    --pause-cost 10.0 \
    --pause-threshold 0.05 \
    --output-dir data/libero/cost_study
```

This will generate:
- `cost_analysis.png`: Visualization with three plots showing:
  - Total cost per chunk and cumulative cost
  - Cost breakdown (execution vs pause)
  - Timing information and gaps
- `cost_summary.txt`: Detailed text summary of the analysis

### Retrospective Analysis (Existing Data)

Analyze costs for episodes that were run without cost analysis:

```bash
# Analyze a single episode
python examples/libero/analyze_costs.py \
    data/libero/sync_5/0/0_libero_10_0_success \
    --execution-cost-rate 1.0 \
    --pause-cost 10.0

# Analyze all episodes in a directory (recursively)
python examples/libero/analyze_costs.py \
    data/libero/sync_5 \
    --batch \
    --execution-cost-rate 1.0 \
    --pause-cost 10.0
```

## Output Interpretation

### Cost Visualization

The generated plot contains three subplots:

1. **Total Cost per Action Chunk**
   - Bar chart: Per-chunk cost
   - Red line: Cumulative cost over time
   - Use to identify high-cost chunks

2. **Cost Breakdown**
   - Stacked bars showing execution cost (bottom) and pause cost (top)
   - Helps understand the source of costs
   - High pause costs indicate poor scheduling

3. **Execution Timing and Gaps**
   - Bars: Execution duration per chunk
   - Red X markers: Gaps to next chunk
   - Orange dashed line: Pause threshold
   - Red highlighting: Chunks where pauses occurred

### Cost Summary

The text summary provides:
- Configuration parameters used
- Overall statistics (total cost, breakdown, number of pauses)
- Per-chunk details including:
  - Timing information
  - Whether chunk was replanned early
  - Cost breakdown

## Use Cases

### 1. Comparing Scheduling Strategies

Run experiments with different broker configurations and compare costs:

```bash
# SyncBroker
python examples/libero/main_multi_robot_runtime.py \
    --enable-cost-analysis \
    --action-chunk-broker.broker-type SYNC \
    --output-dir data/cost_study/sync

# RTCBroker with different parameters
python examples/libero/main_multi_robot_runtime.py \
    --enable-cost-analysis \
    --action-chunk-broker.broker-type RTC \
    --action-chunk-broker.s-min 5 \
    --action-chunk-broker.d-init 4 \
    --output-dir data/cost_study/rtc_5_4
```

### 2. Sensitivity Analysis

Study how different cost parameters affect the optimization:

```bash
# High pause penalty
python examples/libero/analyze_costs.py \
    data/episodes/experiment_1 \
    --pause-cost 50.0 \
    --batch

# Low pause penalty
python examples/libero/analyze_costs.py \
    data/episodes/experiment_1 \
    --pause-cost 1.0 \
    --batch
```

### 3. Identifying Bottlenecks

Look for:
- Chunks with high pause costs → scheduling inefficiencies
- Long execution durations → potential for early replanning
- Frequent early replanning → consider adjusting execution horizons

## Extending the Cost Model

The cost model is designed to be configurable and extensible. To modify the cost calculation:

1. Edit `CostConfig` in `cost_analyzer.py` to add new parameters
2. Update the `analyze()` method in `CostAnalyzer` to implement new cost logic
3. Add corresponding command-line arguments in `main_multi_robot_runtime.py` and `analyze_costs.py`

Example extensions:
- Non-linear cost functions (e.g., quadratic penalties)
- Time-dependent costs (e.g., higher costs during critical phases)
- Latency-based penalties
- Cost based on replanning frequency

## Implementation Details

The cost analyzer:
- Operates independently without affecting execution
- Uses timestamps to track chunk execution timing
- Detects pauses by measuring gaps between consecutive chunks
- Handles early replanning (when actual steps < execution horizon)
- Generates visualizations using matplotlib

Data flow:
1. `Saver` collects timestamps and action chunks during execution
2. `CostAnalyzer.analyze()` processes the data to compute costs
3. Results are saved to disk as plots and text summaries
4. No impact on runtime performance or behavior

## Example Results

After running with cost analysis enabled, each episode directory will contain:

```
0_libero_10_0_success/
├── metadata.json
├── timestamps.csv
├── action_chunks.csv
├── action_chunks.parquet
├── out.mp4
├── debug_data.npz
├── cost_analysis.png          # ← Cost visualization
└── cost_summary.txt           # ← Detailed cost breakdown
```

## Tips for Optimal Scheduling

Based on cost analysis results:

1. **Minimize pauses**: Ensure chunks are requested early enough to avoid gaps
2. **Balance horizon length**: Longer horizons reduce pause frequency but may increase execution cost if replanned
3. **Tune broker parameters**: Use cost analysis to guide RTC broker parameter selection
4. **Consider latency**: Account for model inference time in scheduling decisions
