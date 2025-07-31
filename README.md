
# DDRB-Empis-Lab Investigation Project

## Principal Objective

Create a model that accelerates and diversifies the learning process of classic Super Mario Bros gameplay machine learning models through Procedural Content Generation ML (PCGML) and a variant of Dynamic Difficulty Adjustment. We propose that this adjustment should be considered a reward-difficulty balancing mechanism rather than direct difficulty adjustment.

## Sub-Objectives

1. **Develop a PCGML Model**
   - Create a model that considers latent representations of game levels
   - Incorporate user/agent jump performance descriptions into the generation process

2. **Generate Map Elite Grammar Representations**
   - Produce generative representations of preselected feasible sub-levels
   - Ensure high variety within a fixed expected jump performance space

3. **Implement Reward-Balancing RL Agent**
   - Develop a reinforcement learning agent to balance reward components
   - Distribute rewards strategically across levels to guide user/agent toward expected jump performance

4. **Create Environment Manager**
   - Design a system to control hyperparameters of all previous components
   - Optimize long-term user/agent results with increasing objective performance

5. **Comprehensive Testing and Validation**
   - Test the complete environment with different Super Mario Bros player ML agents
   - Evaluate improvements in original ML agent robustness
   - Measure acceleration in convergence to optimal performance
   
   *Note: We plan to extend testing to real gameplay scenarios using Dynamic Difficulty Reward Balancing trained with multiple artificial agents.*

## Technical Considerations

**Stochastic Reward Definition**: We propose implementing a stochastic reward system where game elements represent multiple potential outcomes. For example, an enemy entity could yield negative rewards if it defeats the player, but positive rewards if the player successfully defeats it first.

## Research Hypothesis

Our model is expected to:
- Improve ML agent robustness in Super Mario Bros gameplay
- Accelerate convergence to optimal performance compared to traditional training methods
- Demonstrate the effectiveness of reward-difficulty balancing over conventional difficulty adjustment

---

**STATUS: DOCUMENT IN PROGRESS**

*This README will be expanded to include detailed methodology, implementation details, experimental design, and results sections as the investigation progresses.*


## Literature Review

### Mario Bros. Generative Framework

#### Mario Bros. Game
*[To be expanded with game mechanics and design principles]*

#### Mario PCG Framework
*[To be expanded with existing procedural content generation approaches]*

#### Grammars and Graphs
*[To be expanded with formal grammar representations and graph-based level generation]*

#### Scene Stitching
*[To be expanded with level assembly and connectivity validation methods]*

#### CUDA and Binary Algebra
*[To be expanded with computational optimization techniques]*

---

### Models

**Note: The following sections represent a collection of approaches not necessarily integrated. Only those marked with † are confirmed for use in the current version.**

#### World Models †

Our overall architecture represents a variation of the [World Model](https://arxiv.org/pdf/1803.10122) framework. Traditional World Models propose a unified model that learns how actions influence an environment while simultaneously learning optimal action selection, grouping interaction and world generation into a single task.

Instead, we propose a modular approach using two separate components: a Player/Agent-Player and a Levels-Generator. This separation maintains two key capabilities:

1. **Player Substitution**: The ability to replace the Player-Agent with real human player interaction
2. **Training Loop Preservation**: Incorporating Asymmetric Adversarial Network advantages as mutual training data generators

This approach sacrifices the Model Dream mechanism but enables better control within search spaces, focusing on agent learning optimization. Our hypothesis is that this separation may improve training speed and/or robustness compared to models trained under traditional conditions.

**Reference**: [General agents need world models](https://arxiv.org/pdf/2506.01622)

#### RL Agent-Player †
*[To be expanded with reinforcement learning agent architecture and training methods]*

#### PCGML for DDA and Reward Balancing †

##### The Flow / The Zone
*[To be expanded with flow state theory and optimal challenge-skill balance]*

##### Dynamic Difficulty Adjustment (DDA)
*[To be expanded with adaptive difficulty mechanisms]*

##### Reward Systems
*[To be expanded with reward structure and balancing methodologies]*

#### Fi-2Pop and Map-Elites †

Diversity is crucial in both evolutionary algorithms and deep learning. Our Algebraic-Powered Search (APS) approach prefilters solvable levels using ML Surrogate Fi-2Pop methods (*Appendix A pending*). However, this may not capture all possible solvable levels, potentially reducing data variability.

To compensate for this inherent limitation, we employ multidimensional Map-Elites over a grid system that guarantees specific types of variability, creating a valuable variability vs. control trade-off.

**Solvability Assessment**: The rules determining level solvability remain under discussion. Current methods generate functional levels but may include false positives/negatives (*Discussion 1 appendix pending*).

**Scene Stitching Variation**: Our current design assumes a level is solvable if every landing tile in the final position has a valid path of connected landings. This axiom allows us to determine solvability by verifying that for all non-empty landing tiles, there exists at least one previous tile with a valid Mario movement connection (jump or walk).

By treating each Mario movement as a validation process rather than a construction process, we can validate random levels by reading from right to left in a single pass. This approach is computationally efficient and parallelizable on GPU through CUDA-supported algebraic operations (*Appendix A pending*).

**Map-Elites Implementation**: We use Map-Elites to guarantee diversity among filtered solvable levels. The approach assigns points across feature functions F: level → [0,1] that describe level characteristics. For example, `scripts\Variety_Domains.py` contains functions that effectively disperse the space of possible levels.

Accumulation points are selected within [0,1], creating a grid across the multidimensional feature function space ∏F_n. Each grid node represents sufficiently different levels while maintaining spatial correlation. Each accumulation point stores the best-performing level in its range, optimizing for both diversity and quality.

The performance function must describe both player-agent jump mechanics performance and associated loss functions, enabling model training with consistent general objectives. This creates an adversarial network where both components optimize the same objective through different metrics and environmental influences.

Current feature functions, evaluations, and loss functions remain under investigation (*Discussion 2 appendix pending*).

#### Mixture of Experts (MOEs)
*[To be expanded with multi-expert model architectures]*

#### Dynamic Chunking and Bit-Networks

These approaches are considered for computational efficiency to achieve real-time generation latency in gameplay, following efficiency model principles:

**References**:
- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/pdf/2310.11453)
- [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/pdf/2507.07955)

#### Sparse Autoencoders (Under Analysis)

**Reference**: [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity?curius=4744)

#### Fine-tuning by Low Rank Adaptations (LoRA)

Considered as an option for exploring local optima, such as user-level fine-tuning applications.
---


### DDA and Reward Balancing

#### The flow / The Zone

#### DDA

#### Reward


---


## Methodology

### Level Representations

We consider two equivalent approaches for level representation: **Grammars** and **Tile Maps**. Both will be treated interchangeably due to their functional equivalence. Grammars are string representations where each token corresponds to a node in a graph called Solvable Generated Spaces Grammars. These grammars provide optimized generation methods that offer better feature control and computational efficiency compared to pure random generation (*Appendix_A pending*).

Each token has a dual representation as tiles. Consider an *n*×*m* grid (matching Mario's display screen size), where each coordinate maps to a vertical tile following the **Super_Mario_Brothers_Maps** standard PCG representation format (*references pending - repository link to be recovered*). The vertical tile data is stored in `Super_Mario_Brothers_Maps\Multi-layer\smb-multi-layer.json`.

Additionally, each level can be represented as:
- A layered binary tensor of dimensions *n*×*m*×*k* (where *k* is the number of features per tile)
- A *k*×*m*-dimensional vector with integers between 0 and (2^*(n+1)*)-1

The integer vector representation is optimal for CUDA implementations, providing acceleration benefits and faster grammar system processing through Algebraic-Powered Search (APS), following scene Stitching principal ([scene stitching]([https://arxiv.org/pdf/2002.02992)) (*Appendix_A pending*) over Solvable Generated Spaces Grammars.

### Gameplay and Model Interactions

Our approach utilizes the `Super_Mario_Brothers_Maps\Multi-layer\smb-multi-layer.json` representation for game development and future agent players. The integer vector format enables computational optimization for solvable level generation and search processes. The PCGML model will induce specific behaviors through Performance-Reward assessment by modifying the search space (i.e., grammars and generative graphs).

The expected data flow is illustrated below, where:
- Intermittent lines (*- - -*) represent the initial data collection phase using only Algebraic-Powered Search (APS)
- Continuous lines (*---*) represent the combined PCGML generation and APS approach

Generated level examples are available in `.txt` and primitive `.png` formats at: `Super_Mario_Brothers_Maps\final_levels\`

![image info](ilustrations\DDRB_high-level_Architecture.png)

*(Detailed graph explanation pending)*

### Current State

Gameplay data collection protocols depend on game design objectives, which remain undefined. The concept encompasses all aspects derived from jump mechanics, including:
- Platform-to-platform navigation
- Enemy elimination (Goombas)
- Coin collection
- Block destruction
- Evasion maneuvers

Currently, we have implemented a Map-Elites system that accepts a **feature function** and collects levels nearest to accumulation points within the **feature function** image space. This system works with any combination of example levels from the 15 available levels in `Super_Mario_Brothers_Maps\Processed\` (including levels 1-1, 1-2, 1-3, and the first level of each world up to World 8). The system accepts any **feature function** that processes a level and returns a value in the range [0,1] (note: open bound variations are also supported for accumulation points). Implementation details are available in `main.py`.

![image info](ilustrations\level_1-1_tiles.png)

### Universal Structures Dataset
The Universal Structures dataset is stored in Super_Mario_Brothers_Maps\structures\Universal_Structures.txt and defines the core building blocks for level generation. Each structure is characterized by multiple feature layers:
```python
index,token,Structures,Landings,Colliders,Reward,Hazard
0,0,-------------X,1,1,0,0
16,1,---------Q---X,17,17,16,0
20,2,---------S---X,17,17,0,0
21,3,---------?--EX,17,17,16,2
22,4,-----Q---S---X,273,273,256,0
28,5,-----------<[X,4,7,0,0
29,6,----------->]X,4,7,0,0
38,7,----------<[[X,8,15,0,0
39,8,---------->]]X,8,15,0,0
41,9,------------EX,1,1,0,2
46,10,---------<[[[X,16,31,0,0
47,11,--------->]]]X,16,31,0,0
64,12,--------S----X,33,33,0,0
```
#### Structure Components:

* Index: Original first plot
* Token: Corresponding grammar token
* Structures: Visual representation of the level structure
* Landings: Numerical encoding for platform/landing areas
* Colliders: Collision detection values for solid elements
* Reward: Point values for collectible items (coins, power-ups, enemies)
* Hazard: Risk values for dangerous elements (enemies, obstacles)

#### Level Encoding Example:
The 1-1 level contains 4 feature layers and is represented as a token sequence:
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,3,4,0, ...]
This sequence maps directly to the Universal Structures dataset, where each token corresponds to a specific structural element with its associated gameplay properties.

The examples of ASG generated level are in Super_Mario_Brothers_Maps\final_levels, e.g.:

![image info](ilustrations\gen_1-1_level.png)

### Next Steps

1. Develop gameplay mechanics
2. Design game features and Performance-Reward data collection systems  
3. Build PCGML model
4. Conduct first round of testing

---

## future sections

**Technical Architecture**

**Experimental Design**

**Implementation Details**

**Results and Analysis**

**Future Work**

**References**