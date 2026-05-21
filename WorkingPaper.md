# DDRB-Empis-Lab Investigation Project

## Abstract

This paper describes a Dynamic Difficulty Reward Balancing (DDRB) pipeline for Super Mario Bros procedural content generation and reinforcement learning. We combine grammar-based level encoding, Map-Elites diversity search, and a solvability filter to produce playable levels for agent training. The core hypothesis is that balancing reward components across generated content can improve agent robustness and convergence compared with fixed-level baselines.

## Research Question

Can Dynamic Difficulty Reward Balancing (DDRB) improve agent robustness and convergence in Super Mario Bros training compared to fixed-level training and equivalent PCGML baselines without reward balancing?

## Contribution Compared to Prior Mario PCG/DDA Work

- Defines DDRB as a reward-weighting mechanism that shapes generation indirectly through training signal balancing, rather than adjusting level difficulty directly.
- Integrates grammar-encoded Super Mario Bros structures with Map-Elites feature coverage and a concrete solvability check.
- Demonstrates an evaluation workflow that compares original SMB baselines with generated level training under DDRB.
- Uses a specialized solvability algorithm in `world_generation/Word_Stitching.py` to filter candidate levels before training.

## Project Objectives

1. Implement a DDRB-aware PCGML generation pipeline for Super Mario Bros.
2. Support level representations as grammar tokens, tile maps, and compact integer encodings.
3. Apply a movement-based solvability validation step to generated candidates.
4. Use Map-Elites feature functions to preserve diverse training content.
5. Evaluate robustness and convergence against baseline agents.

## Experimental Plan

### Agents to Evaluate

- `tabular_q` agent from `mario_rl_env/run_agent.py`
- baseline agent trained on original Super Mario Bros levels
- DDRB-trained agent using validated generated levels
- non-DDRB PCGML agent trained on generated levels without reward balancing (comparison arm)

### Baselines

- fixed-level training on processed original levels in `Super_Mario_Brothers_Maps/Processed/`
- generated-level training with the same PCGML pipeline but without DDRB reward updates
- a structural difficulty adjustment baseline (if available) that modifies level layout instead of reward weights

### Metrics

Robustness:

- completion rate on held-out generated levels
- performance variance across Map-Elites niches
- transfer from original levels to generated level evaluation sets

Convergence:

- episodes required to reach a target average reward threshold
- training curve slope and stability over time
- final policy performance on both original and generated evaluation levels

Secondary metrics:

- feature-space coverage achieved by Map-Elites
- diversity of selected generated levels in landing/collider statistics

### Evaluation Protocol

- train each agent for a fixed budget of episodes
- evaluate on the same validated generated level set and on held-out original levels
- compare means and standard deviations across multiple random seeds
- measure both robustness and convergence improvements relative to baselines

## Methodology

### Level Representation Formats

This project uses three complementary representations:

- **Grammar token sequences** from `Super_Mario_Brothers_Maps/structures/Universal_Structures.txt` for structured construction and pattern reuse.
- **Tile-map encodings** for standard Super Mario Bros level geometry and agent observation states.
- **Compact integer vectors** that encode landing and collider attributes in discrete bit ranges for fast validity checks.

The compact integer encoding stores per-column attributes that are used by solvability and feature extraction routines in `world_generation/Word_Stitching.py` and `world_generation/Variety_Domains.py`.

### Solvability Validation Algorithm

Solvability filtering is implemented in `world_generation/Word_Stitching.py`.

Key components:

- `Punctual_Binary_Prop(n, k)` inspects a specific bit in a tile or column encoding.
- `Range_Binary_Prop(n, a, b)` reads a contiguous bit range for attribute extraction.
- `Jumping_Fiasible_Word(Level, knowledge)` determines if landing columns are reachable by feasible Mario jumps and walks.

Validation criteria:

- valid start and goal landing structures are present
- each landing column can be reached from a prior landing by the agent's movement constraints
- path feasibility is checked conservatively by requiring at least one valid path rather than exhaustive state enumeration

### Feature Functions for Map-Elites

Map-Elites diversity is managed using feature functions in `world_generation/Variety_Domains.py`.

Current features include:

- `calculate_slope(Level, df)` — computes an overall slope from landing heights and normalizes it to (0,1)
- `landings_score(Level, df)` — measures normalized landing tile density
- `colliders_score(Level, df)` — measures normalized collider tile density

These features embed levels into a bounded space and guide elite selection toward diverse, behaviorally distinct content.

### Reward-Balancing Mechanism

Dynamic Difficulty Reward Balancing (DDRB) adjusts reward weights across gameplay signals instead of directly modifying level difficulty.

The mechanism:

- tracks multiple reward components such as traversal success, enemy interaction, progression, and failure penalties
- updates component weights as training proceeds to encourage stable learning dynamics
- influences generation indirectly by selecting levels that match the desired reward profile during Map-Elites search

The `main.py` pipeline applies DDRB via the `Performance` payload and `variety_domains` settings, linking reward-driven generation with Map-Elites selection.

### Dataset and Source Levels

Source data is drawn from both original and generated assets:

- original levels in `Super_Mario_Brothers_Maps/Original/`
- processed training levels in `Super_Mario_Brothers_Maps/Processed/`
- grammar structure definitions in `Super_Mario_Brothers_Maps/structures/Universal_Structures.txt`
- generated candidate levels in `Super_Mario_Brothers_Maps/generated_levels/`

For evaluation, the primary training set is validated generated content, while the baseline set uses processed original levels.

### Generated Levels Validation

Generated candidates are validated in two stages:

1. grammar and token consistency is verified against the universal structure set.
2. playability is checked via `Jumping_Fiasible_Word` in `world_generation/Word_Stitching.py`.

Only levels passing both checks are used for DDRB training and evaluation.

### Prototype Pipeline

The prototype workflow in `main.py` demonstrates:

- creating a generation instance through the API
- requesting `Performance` generation with Map-Elites feature domains and grid points
- running a `tabular_q` agent over the validated generated level pool

This workflow supports direct comparison between DDRB training and baseline agents.

![DDRB architecture](ilustrations/DDRB_high-level_Architecture.png)
*Figure 1: DDRB architecture for PCGML generation, validation, and agent training.*

![Mario level sample](ilustrations/level_1-1_tiles.png)
*Figure 2: Token and tile representation of Super Mario Bros level 1-1.*

![Generated level sample](ilustrations/gen_1-1_level.png)
*Figure 3: Example generated level selected by Map-Elites.*

## Results / Expected Outcomes

Preliminary implementation indicates the DDRB pipeline is operational and that generated levels are filterable with the current solvability heuristics.

Expected outcomes:

- stronger completion performance on generated evaluation levels than fixed-level baselines
- improved convergence speed for DDRB-trained agents compared to non-DDRB PCGML training
- broader Map-Elites feature coverage through validated generated level selection

The current repository supports an end-to-end prototype from generation request to agent rollout, providing a foundation for formal experiments.

## References

- `world_generation/Word_Stitching.py` — solvability validation via bitwise landing/collider analysis
- `world_generation/Variety_Domains.py` — Map-Elites feature functions
- `main.py` — DDRB pipeline orchestration and agent evaluation
- `mario_rl_env/run_agent.py` — tabular Q-learning agent implementation
