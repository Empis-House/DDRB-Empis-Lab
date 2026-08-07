"""
constants.py – All tuneable parameters for the Mario RL environment.
Edit here to adjust physics, screen size, rewards, etc.
"""

# ── Tile geometry ──────────────────────────────────────────────────────────────
TILE_SIZE: int = 64          # pixels per tile (must match your renderer scale)
TILE_SIZE_F: float = float(TILE_SIZE)

# ── Screen / viewport ─────────────────────────────────────────────────────────
SCREEN_TILES_W: int = 16     # tiles visible horizontally
SCREEN_TILES_H: int = 14     # tiles visible vertically  (== map height)
SCREEN_W: int = TILE_SIZE * SCREEN_TILES_W   # 512 px
SCREEN_H: int = TILE_SIZE * SCREEN_TILES_H   # 448 px

# ── Simulation ─────────────────────────────────────────────────────────────────
FPS: int = 60
MAX_STEPS_PER_EPISODE: int = 500000   # truncation limit

# ── Physics ───────────────────────────────────────────────────────────────────
GRAVITY: float          = 0.025*TILE_SIZE_F    # px / frame²
MAX_FALL_SPEED: float   = 0.38*TILE_SIZE_F    # px / frame  (terminal velocity)
MARIO_WALK_SPEED: float = 0.08*TILE_SIZE_F    # px / frame
MARIO_RUN_SPEED: float  = 0.15*TILE_SIZE_F     # px / frame
MARIO_JUMP_VY: float    = -0.5*TILE_SIZE_F   # px / frame  (negative = upward)
BOUNCE_VY: float        = -0.2*TILE_SIZE_F    # when stomping an enemy
GOOMBA_SPEED: float     = 0.038*TILE_SIZE_F     # px / frame

# ── Mario hitbox (fraction of tile) ──────────────────────────────────────────
MARIO_W: int = int(TILE_SIZE * 0.75)   # 24 px
MARIO_H: int = int(TILE_SIZE * 0.90)   # 28 px  (small mario)
MARIO_H_BIG: int = int(TILE_SIZE * 1.85)  # 59 px  (big mario – two tiles tall)

# ── Goomba hitbox ──────────────────────────────────────────────────────────────
GOOMBA_W: int = int(TILE_SIZE * 0.85)   # 27 px
GOOMBA_H: int = int(TILE_SIZE * 0.85)   # 27 px

# ── Rewards ───────────────────────────────────────────────────────────────────
REWARD_PER_TILE_RIGHT: float =  1.0     # reward for each tile advanced right
REWARD_PER_TILE_LEFT:  float = -0.5     # penalty for moving backwards
REWARD_KILL_ENEMY:     float =  5.0
REWARD_COIN:           float =  2.0
REWARD_DEATH:          float = -15.0
REWARD_WIN:            float =  100.0   # reach end of level
REWARD_TIME_STEP:      float = -0.01    # tiny per-step time penalty

# ── Observation tile window (for 'tiles' obs mode) ────────────────────────────
OBS_TILES_W: int = 16    # width of tile window centred on Mario
OBS_TILES_H: int = 14    # height of tile window

# ── Colours (RGB) ─────────────────────────────────────────────────────────────
COL_SKY       = (75, 195, 255)
COL_GROUND    = (140,  80,  20)
COL_BRICK     = (180, 100,  30)
COL_QUESTION  = (250, 200,   0)
COL_QUESTION_USED = (140, 100, 70)
COL_COIN      = (252, 220,  40)
COL_PIPE_DARK = ( 10, 130,  20)
COL_PIPE_LITE = ( 50, 200,  60)
COL_PIPE_RED_DARK = (180, 30, 30)
COL_PIPE_RED_LITE = (220, 70, 70)
COL_PIPE_BLUE_DARK = (25, 70, 170)
COL_PIPE_BLUE_LITE = (65, 125, 235)
COL_MARIO     = (220,  50,  20)
COL_MARIO_HAT = (200,  30,   0)
COL_MARIO_BIG = (200, 130,  60)
COL_GOOMBA    = (120,  70,  10)
COL_GOOMBA_DEAD = (100, 50, 5)
COL_HUD_BG    = (  0,   0,   0)
COL_WHITE     = (255, 255, 255)
COL_YELLOW    = (255, 230,   0)
COL_RED       = (230,  30,  10)
