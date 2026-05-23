# Architecture Proposal: 2D Model/State (MuJoCo-inspired)

## 2.1 Immutable Model (Model2D)
Stores constant data:
- Kinematic tree: `parent` (parent body indices)
- Link transforms: `Xtree` (list of 3×3 SE(2) matrices)
- Joint types: `joint_type` ('revolute' or 'prismatic')
- Inertia: `mass`, `com`, `I_com`
- Gravity: `gravity`
- `floating` flag (if True, first 3 DOF are x, y, theta)

## 2.2 Mutable State (State2D)
Stores time‑varying data:
- `q` – generalized coordinates (joint angles/displacements + optional x, y, theta)
- `qd` – generalized velocities
- Helper methods: `get_base_pose()`, `get_joint_positions()`, `get_joint_velocities()`

## 2.3 Fixed vs Floating Base
- Fixed‑base: `floating=False`, `nq = number of joints`
- Floating‑base: `floating=True`, `nq = 3 + number of joints`

## 3. Interfaces
```python
@dataclass(frozen=True)
class Model2D:
    nq: int
    nv: int
    floating: bool
    parent: np.ndarray
    Xtree: List[np.ndarray]   # 3x3 SE(2) matrices
    joint_type: List[str]
    joint_parent: np.ndarray
    mass: np.ndarray
    com: np.ndarray
    I_com: np.ndarray
    gravity: np.ndarray

class State2D:
    q: np.ndarray
    qd: np.ndarray
    def get_base_pose(self) -> Tuple[float, float, float]
    def get_joint_positions(self) -> np.ndarray
    def get_joint_velocities(self) -> np.ndarray
