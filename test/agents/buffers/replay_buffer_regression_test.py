# Test case following [dopamine buffer_regression_test](https://github.com/google/dopamine/blob/master/tests/dopamine/jax/replay_memory/replay_buffer_regression_test.py)
import unittest 
import numpy as np
import copy
from dotmap import DotMap
from omegaconf import OmegaConf
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from src.agents.buffers.per_buffer import PERBuffer

OBSERVATION_SHAPE = (1, 1, 84, 84)
OBS_DTYPE = np.uint8
STACK_SIZE = 4
BATCH_SIZE = 5
N_STEP = 5
DEVICE = 'cuda'


class ReplayBufferRegressionTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        # Clear existing Hydra instance if already initialized
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        # Get buffer configs
        initialize(version_base=None, config_path='../configs') 
        cfg = compose(config_name='ddqn')
        cfg = DotMap(OmegaConf.to_container(cfg.agent))
        cfg.total_optimize_steps = int((cfg.num_timesteps - cfg.min_buffer_size) * cfg.optimize_per_env_step)
        self.buffer_cfg = cfg.pop('buffer')
        self.buffer_cfg['n_step'] = N_STEP
        buffer_type = self.buffer_cfg.pop('type')
        if 'prior_weight_scheduler' in self.buffer_cfg:
            self.buffer_cfg.prior_weight_scheduler.step_size = cfg.total_optimize_steps
        
        buffer = PERBuffer(device=DEVICE, gamma=cfg['gamma'], **self.buffer_cfg)
        self.memory = buffer

    def testNSteprewards(self):
        memory = copy.deepcopy(self.memory)
        for i in range(50):
            memory.store(
                obs=np.full(OBSERVATION_SHAPE, i, dtype=OBS_DTYPE),
                action=0,
                reward=2.0,
                done=False,
                next_obs=np.full(OBSERVATION_SHAPE, i+1, dtype=OBS_DTYPE)
            )
        
        for _ in range(100):
            batch = memory.sample(BATCH_SIZE)
            np.testing.assert_array_equal(batch['return'].detach().cpu().numpy(),
                                          np.ones(BATCH_SIZE, dtype=np.float32) * 2.0 * ((1-0.99**N_STEP)/(1-0.99)))
            
    def testSampleTransitionBatch(self):
        self.buffer_cfg['n_step'] = 1
        memory = PERBuffer(device=DEVICE, gamma=0.99, **self.buffer_cfg)
        nums_add = 50
        index_to_id = []
        for i in range(nums_add):
            terminal = i % 4 == 0
            memory.store(
                obs=np.full(OBSERVATION_SHAPE, i, dtype=OBS_DTYPE),
                action=0,
                reward=0,
                done=terminal,
                next_obs=np.full(OBSERVATION_SHAPE, i+1, dtype=OBS_DTYPE)
            )
            if not terminal:
                index_to_id.append(i)
        for _ in range(100):
            batch = memory.sample(2)
            self.assertEqual(batch['obs'].shape[0], 2)
        # Check multiple size batch sampling
        for _ in range(100):
            batch = memory.sample(BATCH_SIZE)
            self.assertEqual(batch['obs'].shape[0], BATCH_SIZE)
        
        # Verify we sample the expected indices.
        # Use the same rng state for reproducibility (Not currently implemented.)
        # rng = np.random.default_rng(0)
        batch = memory.sample(BATCH_SIZE)
        indices = [int(state.detach().cpu().numpy()[0,0,0,0,0] * 255.0) 
                   for state in batch["obs"]]
        
        def make_state(key: int) -> np.ndarray:
            return np.full((1,) + OBSERVATION_SHAPE, key, dtype=np.float32) / 255.0
        
        expected_states = np.array([make_state(i) for i in indices])
        expected_next_states = np.array([make_state(i+1) for i in indices])
        expected_terminal = np.array([i % 4 == 0 for i in indices])
        np.testing.assert_equal(batch['obs'].detach().cpu().numpy(), expected_states)
        np.testing.assert_equal(batch['act'].detach().cpu().numpy(), np.zeros((BATCH_SIZE)))
        np.testing.assert_equal(batch['return'].detach().cpu().numpy(), np.zeros((BATCH_SIZE)))
        np.testing.assert_equal(batch['next_obs'].detach().cpu().numpy(), expected_next_states)
        np.testing.assert_equal(batch['done'].detach().cpu().numpy(), expected_terminal)
    
    def testOverFill(self):
        self.buffer_cfg['size'] = 10
        memory = PERBuffer(device=DEVICE, gamma=0.99, **self.buffer_cfg)
        for i in range(36):
            memory.store(
                obs=np.full(OBSERVATION_SHAPE, i, dtype=OBS_DTYPE),
                action=i // 2,
                reward=i-1,
                done= i % 4 == 0,
                next_obs=np.full(OBSERVATION_SHAPE, i+1, dtype=OBS_DTYPE)
            )
        for i in range(2):
            batch = memory.sample(10)
            state_indices = [int(state.detach().cpu().numpy()[0,0,0,0,0] * 255.0) 
                   for state in batch["obs"]]
            idxs = batch['idxs']
            priorities = np.arange(10,110,10)
            print(f"before: {state_indices:};{batch['weights']:}")
            memory.update_priorities(idxs, priorities)
            expected_priorites = np.power(priorities, memory.prior_exp)
            print(f"after: {state_indices:};{expected_priorites:}")

if __name__ == '__main__':
    unittest.main()