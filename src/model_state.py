import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

def se2_transform(x: float, y: float, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, x],
        [s,  c, y],
        [0,  0, 1]
    ])

@dataclass(frozen=True)
class Model2D:
    nq: int
    nv: int
    floating: bool
    parent: np.ndarray          
    Xtree: List[np.ndarray]    
    joint_type: List[str]       
    joint_parent: np.ndarray    
    mass: np.ndarray            
    com: np.ndarray             
    I_com: np.ndarray           
    gravity: np.ndarray         

    @classmethod
    def from_twolink(cls):
        parent = np.array([-1, 0])
        Xtree = [
            se2_transform(0, 0, 0),   
            se2_transform(1, 0, 0)   
        ]
        joint_type = ['revolute', 'revolute']
        joint_parent = np.array([0, 1])
        mass = np.array([1.0, 1.0])
       
        com = np.array([[0.5, 0.0], [0.5, 0.0]])
        I_com = np.array([0.01, 0.01])   
        gravity = np.array([0.0, -9.81])
        return cls(
            nq=2, nv=2, floating=False,
            parent=parent, Xtree=Xtree,
            joint_type=joint_type, joint_parent=joint_parent,
            mass=mass, com=com, I_com=I_com,
            gravity=gravity
        )

    @classmethod
    def from_cartpole(cls):
        parent = np.array([-1, 0])
        Xtree = [
            se2_transform(0, 0, 0),   
            se2_transform(0, 0, 0)   
        ]
        joint_type = ['prismatic', 'revolute']
        joint_parent = np.array([0, 1])
        mass = np.array([1.0, 0.1])
        com = np.array([[0.0, 0.0], [0.5, 0.0]])
        I_com = np.array([0.01, 0.001])
        gravity = np.array([0.0, -9.81])
        return cls(
            nq=2, nv=2, floating=False,
            parent=parent, Xtree=Xtree,
            joint_type=joint_type, joint_parent=joint_parent,
            mass=mass, com=com, I_com=I_com,
            gravity=gravity
        )

    @classmethod
    def from_floating_base(cls, n_joints=2):
        """Floating base + n_joints revolute joints (like a planar robot on a boat)."""
        floating_dof = 3
        total_dof = floating_dof + n_joints
        parent = np.array([-1] + [0]*n_joints)
        Xtree = [se2_transform(0,0,0)] + [se2_transform(0.5,0,0) for _ in range(n_joints)]
        joint_type = ['revolute'] * n_joints
        joint_parent = np.arange(1, n_joints+1)
        mass = np.array([1.0] + [0.5]*n_joints)
        com = np.array([[0,0]] + [[0.5,0]]*n_joints)
        I_com = np.array([0.1] + [0.01]*n_joints)
        gravity = np.array([0, -9.81])
        return cls(
            nq=total_dof, nv=total_dof, floating=True,
            parent=parent, Xtree=Xtree,
            joint_type=joint_type, joint_parent=joint_parent,
            mass=mass, com=com, I_com=I_com,
            gravity=gravity
        )

class State2D:
    def __init__(self, model: Model2D):
        self.model = model
        self.q = np.zeros(model.nq)
        self.qd = np.zeros(model.nv)

    def get_base_pose(self) -> Tuple[float, float, float]:
        if self.model.floating:
            return self.q[0], self.q[1], self.q[2]
        return 0.0, 0.0, 0.0

    def get_joint_positions(self) -> np.ndarray:
        start = 3 if self.model.floating else 0
        return self.q[start:]

    def get_joint_velocities(self) -> np.ndarray:
        start = 3 if self.model.floating else 0
        return self.qd[start:]
    
def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

if __name__ == "__main__":
    model1 = Model2D.from_twolink()
    
    parent_ref = np.array([-1, 0])
    Xtree_ref = [se2_transform(0,0,0), se2_transform(1,0,0)]
    joint_type_ref = ['revolute', 'revolute']
    joint_parent_ref = np.array([0, 1])
    mass_ref = np.array([1.0, 1.0])
    com_ref = np.array([[0.5, 0.0], [0.5, 0.0]])
    I_com_ref = np.array([0.01, 0.01])
    gravity_ref = np.array([0.0, -9.81])
    nq_ref = 2
    nv_ref = 2
    floating_ref = False
    
    assert model1.nq == nq_ref
    assert model1.nv == nv_ref
    assert model1.floating == floating_ref
    assert np.array_equal(model1.parent, parent_ref)
    assert model1.joint_type == joint_type_ref
    assert np.array_equal(model1.joint_parent, joint_parent_ref)
    assert np.allclose(model1.mass, mass_ref)
    assert np.allclose(model1.com, com_ref)
    assert np.allclose(model1.I_com, I_com_ref)
    assert np.allclose(model1.gravity, gravity_ref)
    
    for i, (m, ref) in enumerate(zip(model1.Xtree, Xtree_ref)):
        assert np.allclose(m, ref), f"Xtree[{i}] does not match"
    
    print("The models match – the test is passed.")
    