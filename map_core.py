import networkx as nx
import numpy as np
import logging
from typing import Dict

# [SCI Integration] Import Edge Intelligence Agent
from infrastructure import EdgeSwitchAgent

logger = logging.getLogger("MapCore")


class SpatialFieldGenerator:
    """
    [Environment Modeling]
    Generates spatially correlated environmental factors (Mud Depth).
    Simulates 'Unstructured Environments' where terrain conditions vary continuously.
    """

    def __init__(self, width, height, seed=42):
        np.random.seed(seed)
        self.width = width
        self.height = height
        self.grid = self._generate_correlated_field()

    def _generate_correlated_field(self):
        """
        Uses 2D Gaussian Kernel convolution to simulate continuous fields (e.g., Rice Paddies).
        """
        # 1. White Noise Base
        noise = np.random.rand(self.width, self.height)

        # 2. Gaussian Filter (Smoothing)
        x = np.arange(-2, 3)
        y = np.arange(-2, 3)
        xx, yy = np.meshgrid(x, y)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / 2.0)
        kernel = kernel / np.sum(kernel)

        try:
            from scipy.signal import convolve2d
            field = convolve2d(noise, kernel, mode='same', boundary='symm')
        except ImportError:
            # Fallback if scipy is missing
            field = noise

        # Normalize to [0, 1]
        field = (field - field.min()) / (field.max() - field.min())

        # [Tuning] Bias towards muddy for stress testing
        field = np.clip(field + 0.1, 0.0, 1.0)
        return field

    def get_value_at(self, x, y, max_dim_x, max_dim_y):
        """Map physical coordinates (meters) to grid index."""
        idx_x = int((x / max_dim_x) * (self.width - 1))
        idx_y = int((y / max_dim_y) * (self.height - 1))
        idx_x = np.clip(idx_x, 0, self.width - 1)
        idx_y = np.clip(idx_y, 0, self.height - 1)
        return self.grid[idx_x, idx_y]


class NodeObject:
    def __init__(self, node_id, pos, env_config, has_switch=False):
        self.id = node_id
        self.pos = pos
        self.agent = None

        # [Distributed Deployment]
        # Deploy EdgeSwitchAgent at junctions to enable local control.
        if has_switch:
            # Each switch perceives its LOCAL environment (Mud factor)
            # This supports the "Distributed Perception" claim in the paper.
            local_env = env_config.copy()
            self.agent = EdgeSwitchAgent(node_id, local_env)


class GridMap:
    """
    [Cyber-Physical Topology]
    Integrates the Graph (Cyber) with the Environmental Field (Physical).
    """

    def __init__(self, config):
        self.cfg = config
        self.rows = config['topology']['rows']
        self.cols = config['topology']['cols']
        self.spacing = config['topology']['cell_spacing']

        self.graph = nx.DiGraph()
        self.nodes: Dict[str, NodeObject] = {}

        # Initialize Environmental Field (2km x 2km area)
        self.field_gen = SpatialFieldGenerator(20, 20)
        self.max_dim = max(self.rows, self.cols) * self.spacing

        self._build_topology()
        logger.info(f"Map Initialized: {len(self.nodes)} nodes, Unstructured Mud Field Active.")

    def _build_topology(self):
        # 1. Backbone Grid
        for r in range(self.rows):
            for c in range(self.cols):
                nid = f"N_{r}_{c}"
                x, y = c * self.spacing, r * self.spacing

                # Sample local mud factor from the continuous field
                local_mud = self.field_gen.get_value_at(x, y, self.max_dim, self.max_dim)
                node_env = self.cfg['environment'].copy()
                node_env['mud_factor'] = local_mud

                # Deploy Switch Agent
                self._add_node(nid, (x, y), node_env, has_switch=True)

        # 2. Edges & Intermediate Stops
        for r in range(self.rows):
            for c in range(self.cols):
                u = f"N_{r}_{c}"

                # Horizontal Connections with intermediate stops
                if c < self.cols - 1:
                    v = f"N_{r}_{c + 1}"
                    stop_id = f"Stop_H_{r}_{c}"
                    mid_pos = ((self.nodes[u].pos[0] + (c + 1) * self.spacing) / 2, self.nodes[u].pos[1])

                    # Stops are dumb nodes (no switch agent)
                    mid_mud = self.field_gen.get_value_at(mid_pos[0], mid_pos[1], self.max_dim, self.max_dim)
                    mid_env = self.cfg['environment'].copy()
                    mid_env['mud_factor'] = mid_mud

                    self._add_node(stop_id, mid_pos, mid_env, has_switch=False)

                    self._add_edge(u, stop_id)
                    self._add_edge(stop_id, v)
                    self._add_edge(v, stop_id)
                    self._add_edge(stop_id, u)

                # Vertical Connections
                if r < self.rows - 1:
                    v = f"N_{r + 1}_{c}"
                    self._add_edge(u, v)
                    self._add_edge(v, u)

        # 3. Depots (Connect to nearest backbone node)
        depots = self.cfg['topology'].get('depots', {})
        for name, coords in depots.items():
            # Depots have default environment
            d_env = self.cfg['environment'].copy()
            self._add_node(name, tuple(coords), d_env, has_switch=False)

            # Find nearest backbone node
            candidates = [n for n in self.nodes.keys() if n.startswith("N_")]
            if candidates:
                closest = min(candidates, key=lambda n: np.linalg.norm(np.array(self.nodes[n].pos) - np.array(coords)))
                self._add_edge(name, closest)
                self._add_edge(closest, name)

    def _add_node(self, nid, pos, env_config, has_switch):
        node_obj = NodeObject(nid, pos, env_config, has_switch)
        self.nodes[nid] = node_obj
        self.graph.add_node(nid, pos=pos)

    def _add_edge(self, u, v):
        p1 = np.array(self.nodes[u].pos)
        p2 = np.array(self.nodes[v].pos)
        dist = np.linalg.norm(p1 - p2)

        # Edge properties inherit from connected nodes (Average Mud)
        m1 = self.nodes[u].agent.mud if self.nodes[u].agent else self.nodes[u].pos[0] * 0  # Dummy fallback
        # Better fallback: query field generator again or store mud in NodeObject
        # For simplicity here, we assume node objects hold environmental data implicitly
        # (In full logic, NodeObject would store 'mud' directly)

        self.graph.add_edge(u, v, weight=dist, length=dist)

    def update_infrastructure(self, dt, time):
        """
        [Main Loop Hook]
        Updates all distributed edge agents.
        """
        for node in self.nodes.values():
            if node.agent:
                node.agent.update(dt, time)