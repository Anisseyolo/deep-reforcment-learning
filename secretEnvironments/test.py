import ctypes
import platform
import numpy as np
from models.monte_carlo_es import naive_monte_carlo_with_exploring_starts

if platform.system().lower() == "windows":
    lib_path = "./libs/secret_envs.dll"
elif platform.system().lower() == "linux":
    lib_path = "./libs/libsecret_envs.so"
elif platform.system().lower() == "darwin":
    if "intel" in platform.processor().lower():
        lib_path = "./libs/libsecret_envs_intel_macos.dylib"
    else:
        lib_path = "./libs/libsecret_envs.dylib"


class SecretEnv0Wrapper:
    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self._init_mdp_functions()
        self._init_mc_td_methods()

    def _init_mdp_functions(self):
        self.lib.secret_env_0_num_states.argtypes = []
        self.lib.secret_env_0_num_states.restype = ctypes.c_size_t
        self.lib.secret_env_0_num_actions.argtypes = []
        self.lib.secret_env_0_num_actions.restype = ctypes.c_size_t
        self.lib.secret_env_0_num_rewards.argtypes = []
        self.lib.secret_env_0_num_rewards.restype = ctypes.c_size_t
        self.lib.secret_env_0_reward.argtypes = []
        self.lib.secret_env_0_reward.restype = ctypes.c_float
        self.lib.secret_env_0_transition_probability.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                                                                 ctypes.c_size_t]
        self.lib.secret_env_0_transition_probability.restype = ctypes.c_float

    def _init_mc_td_methods(self):
        self.lib.secret_env_0_new.argtypes = []
        self.lib.secret_env_0_new.restype = ctypes.c_void_p
        self.lib.secret_env_0_reset.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_0_reset.restype = None
        self.lib.secret_env_0_display.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_0_display.restype = None
        self.lib.secret_env_0_state_id.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_0_state_id.restype = ctypes.c_size_t
        self.lib.secret_env_0_is_forbidden.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.secret_env_0_is_forbidden.restype = ctypes.c_bool
        self.lib.secret_env_0_is_game_over.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_0_is_game_over.restype = ctypes.c_bool
        self.lib.secret_env_0_available_actions.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_0_available_actions.restype = ctypes.POINTER(ctypes.c_size_t)
        self.lib.secret_env_0_available_actions_len.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_0_available_actions_len.restype = ctypes.c_size_t
        self.lib.secret_env_0_available_actions_delete.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
        self.lib.secret_env_0_available_actions_delete.restype = None
        self.lib.secret_env_0_step.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.secret_env_0_step.restype = None
        self.lib.secret_env_0_score.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_0_score.restype = ctypes.c_float
        self.lib.secret_env_0_delete.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_0_delete.restype = None
        self.lib.secret_env_0_from_random_state.argtypes = []
        self.lib.secret_env_0_from_random_state.restype = ctypes.c_void_p


class SecretEnv0:
    def __init__(self, wrapper=None, instance=None):
        if wrapper is None:
            wrapper = SecretEnv0Wrapper()
        self.wrapper = wrapper
        if instance is None:
            instance = self.wrapper.lib.secret_env_0_new()
        self.instance = instance

    def __del__(self):
        if self.wrapper is not None:
            self.wrapper.lib.secret_env_0_delete(self.instance)

    # Méthodes MDP
    def num_states(self) -> int:
        return int(self.wrapper.lib.secret_env_0_num_states())

    def num_actions(self) -> int:
        return int(self.wrapper.lib.secret_env_0_num_actions())

    def num_rewards(self) -> int:
        return int(self.wrapper.lib.secret_env_0_num_rewards())

    def reward(self, i: int) -> float:
        return self.wrapper.lib.secret_env_0_reward(i)

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        return self.wrapper.lib.secret_env_0_transition_probability(s, a, s_p, r_index)

    # Méthodes Monte Carlo et TD
    def state_id(self) -> int:
        return int(self.wrapper.lib.secret_env_0_state_id(self.instance))

    def reset(self):
        self.wrapper.lib.secret_env_0_reset(self.instance)

    def display(self):
        self.wrapper.lib.secret_env_0_display(self.instance)

    def is_forbidden(self, action: int) -> int:
        return self.wrapper.lib.secret_env_0_is_forbidden(self.instance, action)

    def is_game_over(self) -> bool:
        return self.wrapper.lib.secret_env_0_is_game_over(self.instance)

    def available_actions(self) -> np.ndarray:
        actions_len = int(self.wrapper.lib.secret_env_0_available_actions_len(self.instance))
        actions_pointer = self.wrapper.lib.secret_env_0_available_actions(self.instance)
        arr = np.ctypeslib.as_array(actions_pointer, (actions_len,))
        arr_copy = np.copy(arr)
        self.wrapper.lib.secret_env_0_available_actions_delete(actions_pointer, actions_len)
        return arr_copy

    def step(self, action: int):
        self.wrapper.lib.secret_env_0_step(self.instance, action)

    def score(self):
        return self.wrapper.lib.secret_env_0_score(self.instance)

    @staticmethod
    def from_random_state() -> 'SecretEnv0':
        wrapper = SecretEnv0Wrapper()
        instance = wrapper.lib.secret_env_0_from_random_state()
        return SecretEnv0(wrapper, instance)


if __name__ == "__main__":
    env = SecretEnv0()
    print(env.num_states())
    print(env.num_actions())
    print(env.num_rewards())
    for i in range(env.num_rewards()):
        print(env.reward(i))
    print(env.p(0, 0, 0, 0))

    print(env.available_actions())

    while not env.is_game_over():
        env.display()
        env.step(env.available_actions()[0])
    env.display()

    print(env.score())

    random_state_env = SecretEnv0.from_random_state()
    print(random_state_env.available_actions())

    env = SecretEnv0()
    num_episodes = 1000
    gamma = 0.9

    num_episodes = int(num_episodes)

    policy, value_function = naive_monte_carlo_with_exploring_starts(env, num_episodes, gamma)
    print("Policy:", policy)
    print("Value Function:", value_function)
