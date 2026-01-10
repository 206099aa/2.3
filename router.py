import networkx as nx
import logging
import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# [SCI Integration] Import High-Fidelity Physics
from physics import DavisResistanceModel

# 配置日志
logger = logging.getLogger("Distributed.Router")


# =========================================================================
# [Layer 1] Stochastic Gossip Channel (Routing Control Plane)
# -------------------------------------------------------------------------
# Simulates the control plane messaging for routing table exchanges.
# distinct from the data plane channel in comms.py.
# =========================================================================

@dataclass
class Packet:
    source_id: str
    target_id: str
    payload: dict
    timestamp: float
    retry_count: int = 0


class GossipChannel:
    """
    [Network Layer]
    Simulates the specific characteristics of routing control messages (Small payloads).
    Uses a probabilistic model to simulate AoI (Age of Information) decay.
    """

    def __init__(self, loss_rate=0.1, avg_delay=0.05):
        self.loss_rate = loss_rate
        self.avg_delay = avg_delay

    def transmit(self, packet: Packet, global_time: float) -> Optional[Packet]:
        # 1. Probabilistic Loss
        if random.random() < self.loss_rate:
            return None

        # 2. Stochastic Delay (Jitter)
        jitter = np.random.exponential(self.avg_delay)
        packet.timestamp = global_time + jitter
        return packet


# =========================================================================
# [Layer 2] Physics-Aware Link Evaluator (Methodology Core)
# -------------------------------------------------------------------------
# Implements the CPS Routing Cost Function J.
# J = alpha * Time + beta * Energy(Physics) + gamma * Risk(Health)
# =========================================================================

class KinodynamicLinkEvaluator:
    """
    [Algorithm] Physics-Aware Cost Estimator.
    Uses Davis Formula and Soil Mechanics to predict edge traversal cost.
    """

    def __init__(self, grid_map, weights=None):
        self.grid = grid_map
        # Import Physics Model
        self.davis_model = DavisResistanceModel()

        # [Sensitivity Analysis Params]
        if weights:
            self.alpha_t = weights.get('alpha_t', 1.0)
            self.beta_e = weights.get('beta_e', 0.1)  # Energy weight
            self.gamma_r = weights.get('gamma_r', 500.0)  # Safety weight
        else:
            self.alpha_t = 1.0
            self.beta_e = 0.1
            self.gamma_r = 500.0

    def evaluate_link(self, u, v, current_mud):
        """
        Calculate Generalized Cost J(u, v).
        """
        # 1. Extract Geometry
        edge_data = self.grid.graph[u][v]
        dist = edge_data.get('weight', 100.0)
        # Local sensing of environment (Distributed perception)
        mud = edge_data.get('mud', current_mud)

        # 2. Physics-based Energy Prediction (The "Cyber-Physical" Link)
        # Estimate traversing velocity based on environmental constraints
        v_max = 15.0  # Limit for Hauler
        v_est = v_max * (1.0 - 0.6 * mud)  # Slow down in mud
        v_est = max(1.0, v_est)

        # A. Davis Resistance (Rolling + Aero)
        # Assume standard mass 5000kg for routing estimation
        f_davis = self.davis_model.compute_resistance(mass_kg=5000.0, velocity=v_est)

        # B. Soil Mechanics Resistance (Sinkage Drag)
        # F_soil ~ k * z^n (Bekker's theory simplified linear approx)
        f_soil = 5000.0 * 9.81 * (0.05 * mud)

        total_force = f_davis + f_soil
        predicted_energy = total_force * dist  # Work = Force * Distance

        # 3. Time Cost
        predicted_time = dist / v_est

        # 4. Risk Cost (Node Health from Edge Agents)
        node_v = self.grid.nodes[v]
        risk_cost = 0.0

        if node_v.agent:
            # 读取边缘节点的健康状态
            h_index = node_v.agent.health_index

            # If switch is STALLED, cost is infinite (Topology Cut)
            if node_v.agent.state.name == 'STALLED':
                return float('inf')

            # Exponential Barrier Function for Reliability
            # Cost explodes as health approaches 0
            risk_cost = self.gamma_r * np.exp(3.0 * (1.0 - h_index))

        # 5. Total Generalized Cost
        J = (self.alpha_t * predicted_time +
             self.beta_e * predicted_energy +
             risk_cost)

        return J


# =========================================================================
# [Layer 3] Distributed Gossip Protocol (Routing Algorithm)
# -------------------------------------------------------------------------
# Implements Distance Vector routing with AoI-based updates.
# =========================================================================

@dataclass
class RoutingEntry:
    next_hop: str
    cost: float
    timestamp: float  # Age of Information (AoI)


class DistributedProtocolSim:
    """
    [Distributed System]
    Simulates asynchronous Bellman-Ford (Distance Vector) over Gossip.
    """

    def __init__(self, grid_map):
        self.grid = grid_map
        self.graph = grid_map.graph

        # Distributed Routing Tables: Node -> {Target -> Entry}
        self.node_tables: Dict[str, Dict[str, RoutingEntry]] = {
            n: {} for n in self.grid.nodes
        }

        self.channel = GossipChannel()
        self.evaluator = KinodynamicLinkEvaluator(grid_map)
        self.last_gossip_time = 0.0

    def gossip_step(self, global_time):
        """
        Execute one round of asynchronous gossip.
        """
        # Frequency Control (e.g. 2Hz)
        if global_time - self.last_gossip_time < 0.5:
            return
        self.last_gossip_time = global_time

        # 1. Random Wake-up (Asynchronous Behavior)
        # In a real distributed system, nodes define their own clock.
        active_ratio = 0.3
        active_nodes = random.sample(list(self.grid.nodes), k=int(len(self.grid.nodes) * active_ratio))

        for u in active_nodes:
            self._broadcast_from_node(u, global_time)
            self._prune_stale_entries(u, global_time)

    def _broadcast_from_node(self, u, now):
        """Node u shares its knowledge with neighbors."""
        if u not in self.node_tables: return

        # Prepare Distance Vector
        payload = {
            tgt: (entry.cost, entry.timestamp)
            for tgt, entry in self.node_tables[u].items()
        }

        for v in self.graph.neighbors(u):
            pkt = Packet(source_id=u, target_id=v, payload=payload, timestamp=now)

            # Channel Transmission
            recv_pkt = self.channel.transmit(pkt, now)

            if recv_pkt:
                self._update_table(v, recv_pkt, now)

    def _update_table(self, receiver, packet, now):
        """
        [Bellman-Ford Relaxation with AoI]
        D(v, t) = min( D(v, t), Cost(v, u) + D(u, t) )
        """
        sender = packet.source_id
        sender_vector = packet.payload
        my_table = self.node_tables[receiver]

        # 1. Evaluate Local Link Cost (Physics-Aware)
        # Use local sensor reading for mud
        local_mud = self.grid.nodes[receiver].agent.mud if self.grid.nodes[receiver].agent else 0.5
        link_cost = self.evaluator.evaluate_link(receiver, sender, local_mud)

        if link_cost == float('inf'): return

        # 2. Relax Edges
        for target, (remote_cost, remote_ts) in sender_vector.items():
            if target == receiver: continue

            total_cost = link_cost + remote_cost

            # AoI Logic: Trust fresher info, or significantly better paths
            # This prevents counting to infinity loops in dynamic graphs
            current_entry = my_table.get(target)

            update = False
            if current_entry is None:
                update = True
            else:
                # If new path is cheaper OR current info is too old
                if total_cost < current_entry.cost:
                    update = True
                elif (remote_ts - current_entry.timestamp) > 5.0:
                    update = True  # Refresh stale info

            if update:
                my_table[target] = RoutingEntry(
                    next_hop=sender,
                    cost=total_cost,
                    timestamp=max(remote_ts, now)
                )

    def _prune_stale_entries(self, u, now):
        """Remove unreachable targets (TTL)."""
        ttl = 30.0
        keys_to_del = []
        for target, entry in self.node_tables[u].items():
            if (now - entry.timestamp) > ttl:
                keys_to_del.append(target)

        for k in keys_to_del:
            del self.node_tables[u][k]

    def inject_destination(self, dest_node, now):
        """Seed the network with a destination."""
        if dest_node in self.node_tables:
            self.node_tables[dest_node][dest_node] = RoutingEntry(
                next_hop=dest_node, cost=0.0, timestamp=now
            )

    def get_next_hop(self, current_node, target_node):
        """Edge API: Get guidance."""
        entry = self.node_tables.get(current_node, {}).get(target_node)
        return entry.next_hop if entry else None


# =========================================================================
# [Layer 4] Intelligent Router Interface
# -------------------------------------------------------------------------
# Unified API for vehicle agents.
# =========================================================================

class IntelligentRouter:
    def __init__(self, grid_map):
        self.protocol = DistributedProtocolSim(grid_map)
        self.fallback_graph = grid_map.graph  # For initial cold start
        self.evaluator = self.protocol.evaluator

    def step(self, global_time):
        self.protocol.gossip_step(global_time)

    def advertise_destinations(self, targets: List[str], global_time: float):
        for t in targets:
            self.protocol.inject_destination(t, global_time)

    def get_dynamic_path(self, start, end, vehicle):
        """
        Hybrid Routing Strategy.
        Prefer Distributed Guidance, Fallback to A* if cold start.
        """
        # 1. Try Distributed Table (Edge Logic)
        hop = self.protocol.get_next_hop(start, end)
        if hop:
            return [start, hop]

        # 2. Fallback A* (Cloud/Onboard Logic) - Only for initialization
        try:
            # Use the same Physics-Aware Cost for consistency
            def cost_func(u, v, d):
                return self.evaluator.evaluate_link(u, v, d.get('mud', 0.5))

            path = nx.astar_path(
                self.fallback_graph, start, end,
                heuristic=lambda u, v: 0,
                weight=cost_func
            )
            return path
        except:
            return []