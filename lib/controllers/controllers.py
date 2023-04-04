import numpy as np
import do_mpc
import casadi as cs


from abc import ABC, abstractmethod
from typing import Set, List, Dict, Optional, Tuple



class AbstractController(ABC):

    def __init__(self):
        self._goal = None

    @abstractmethod
    def step(self, robot_state: np.ndarray, ped_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError()

    @property
    def goal(self) -> Optional[np.ndarray]:
        return self._goal

    @goal.setter
    def goal(self, value: np.ndarray):
        self._goal = value.copy()


class DoMPCController(AbstractController):

    def __init__(self,
                 horizon: int,
                 dt: float,
                 total_peds: int,
                 Q: np.ndarray,
                 R: np.ndarray,
                 W: float,
                 lb: Tuple[float, float],
                 ub: Tuple[float, float],
                 r_robot: float,
                 r_ped: float,
                 dummy_ped: Tuple[float, float],
                 solver: str,
                 cost_type: str,
                 constraint: Optional[Tuple[str, float]]) -> None:
        """Initiazition of the class instance
        Args:
            init_state (np.ndarray): Initial state of the system
            goal (np.ndarray): Goal of the system
            horizon (int): Prediction horizon of the controller, [steps]
            dt (float): Time delta, [s]
            model_type (str): Type of the model, ["unicycle", "unicycle_double_integrator"]
            total_peds (int): Amount of pedestrians in the system, >= 0
            Q (np.ndarray): State weighted matrix, [state_dim * state_dim]
            R (np.ndarray): Control weighted matrix, [control_dim * control_dim]
            lb (List[float]): Lower boundaries for system states or/and controls
            ub (List[float]): Upper boundaries for system states or/and controls
            r_rob (float): Robot radius, [m]
            r_ped (float): Pedestrian radius, [m]
            constraint_value (float): Minimal safe distance between robot and pedestrian, [m]
            predictor (AbstractPredictor): Predictor, [Constant Velocity Predictor, Neural Predictor]
            # TODO
        Returns:
            _type_: None
        """
        super(DoMPCController, self).__init__()

        # Architecture requires at least one dummy pedestrian in the system
        if total_peds == 0:
            total_peds = 1
        self._total_peds = total_peds
        self._dummy_ped = dummy_ped

        # System model
        self._model = do_mpc.model.Model("discrete")

        # State variables
        x = self._model.set_variable("_x", "x")
        y = self._model.set_variable("_x", "y")
        phi = self._model.set_variable("_x", "phi")
        v = self._model.set_variable("_u", "v")
        w = self._model.set_variable("_u", "w")

        # Discrete equations of motion
        self._model.set_rhs("x", x + v * cs.cos(phi) * dt)
        self._model.set_rhs("y", y + v * cs.sin(phi) * dt)
        self._model.set_rhs("phi", phi + w * dt)

        # Pedestrians
        self._state_peds = self._model.set_variable("_tvp", "p_peds", shape=(4, total_peds))
        # Pedestrians' covariances
        # Each covariance matrix is represented a row [a1, a2, b1, b2]
        self._covariances_peds = self._model.set_variable("_tvp", "cov_peds", shape=(4, total_peds))
        # Inverse of covariances
        # Each covariance matrix is represented a row [a1, a2, b1, b2]
        self._inverse_covariances_peds = self._model.set_variable("_tvp", "inv_cov_peds", shape=(4, total_peds))
        # Goal
        current_goal = self._model.set_variable("_tvp", "current_goal", shape=(3,))
        # Initial position on the prediction step
        p_rob_0 = self._model.set_variable("_tvp", "p_rob_0", shape=(3,))

        self._model.setup()

        # MPC controller
        self._mpc = do_mpc.controller.MPC(self._model)
        # Controller setup
        setup_mpc = {
            "n_horizon": horizon,
            "t_step": dt,
            "store_full_solution": True,
            "nlpsol_opts": {"ipopt.print_level": 0,
                            "ipopt.sb": "yes",
                            "print_time": 0,
                            'ipopt.linear_solver': solver}
        }
        self._mpc.set_param(**setup_mpc)

        # Horizontally concatenated array of robot current position for all the pedestrians
        p_robot_hcat = cs.hcat([self._model._x.cat[:2] for _ in range(total_peds)])
        p_peds = self._state_peds[:2, :]
        u = self._model._u.cat

        # stage cost
        u = self._model._u.cat

        if cost_type == "mahalanobis":
            S = 0
            delta = (p_robot_hcat - p_peds)
            for ped_ind in range(total_peds):
                S += 1 / (delta[:, ped_ind].T @ cs.reshape(
                    self._inverse_covariances_peds[:, ped_ind], 2, 2) @ delta[:, ped_ind])
            stage_cost = u.T @ R @ u + W * S

        elif cost_type == "euclidean":
            S = 0
            delta = (p_robot_hcat - p_peds)
            for ped_ind in range(self._total_peds):
                S += 1 / (delta[:, ped_ind].T @ delta[:, ped_ind])
            stage_cost = u.T @ R @ u + W * S

        elif cost_type == "simple":
            stage_cost = u.T @ R @ u
        else:
            raise ValueError(f"Unknown cost type {cost_type}")

        # Goal cost
        p_rob_N = self._model._x.cat[:3]
        delta_p = cs.norm_2(p_rob_N[:2] - current_goal[:2]) / cs.norm_2(p_rob_0[:2] - current_goal[:2])
        goal_cost = Q * delta_p ** 2  # GO-MPC Cost-function

        # Set cost
        self._mpc.set_objective(lterm=stage_cost + goal_cost,
                                mterm=goal_cost)

        self._mpc.set_rterm(v=1e-2, w=1e-2)

        # Bounds
        self._mpc.bounds['lower', '_u', 'v'] = lb[0]
        self._mpc.bounds['lower', '_u', 'w'] = lb[1]
        self._mpc.bounds['upper', '_u', 'v'] = ub[0]
        self._mpc.bounds['upper', '_u', 'w'] = ub[1]

        if constraint is not None:
            constraint_type, constraint_value = constraint
            if constraint_type == "euclidean":
                lb_dists_square = np.array([(r_robot + r_ped + constraint_value) ** 2 for _ in range(total_peds)])
                dx_dy_square = (p_robot_hcat - p_peds) ** 2
                pedestrians_distances_squared = dx_dy_square[0, :] + dx_dy_square[1, :]
                self._mpc.set_nl_cons("euclidean_dist_to_peds", -pedestrians_distances_squared,
                                      ub=-lb_dists_square)

            elif constraint_type == "mahalanobis":
                delta = (p_robot_hcat - p_peds)
                array_mahalanobis_distances = cs.SX(1, total_peds)
                for ped_ind in range(self._total_peds):
                    array_mahalanobis_distances[ped_ind] = delta[:, ped_ind].T @ cs.reshape(
                        self._inverse_covariances_peds[:, ped_ind], 2, 2) @ delta[:, ped_ind]

                array_mahalanobis_bounds = cs.SX(1, total_peds)
                V_s = np.pi * (r_robot + r_ped + 1) ** 2
                for ped_ind in range(self._total_peds):
                    deter = 2 * np.pi * DoMPCController._determinant(
                        cs.reshape(self._covariances_peds[:, ped_ind], 2, 2))
                    deter = cs.sqrt(deter)
                    array_mahalanobis_bounds[ped_ind] = 2 * cs.log(deter * constraint_value / V_s)

                self._mpc.set_nl_cons("mahalanobis_dist_to_peds",
                                      -array_mahalanobis_distances - array_mahalanobis_bounds,
                                      ub=np.zeros(total_peds))

            else:
                raise ValueError(f"Unknown constraint type {constraint_type}")

        # Time-variable-parameter function for pedestrian
        self._mpc_tvp_fun = self._mpc.get_tvp_template()
        def mpc_tvp_fun(t_now):
            return self._mpc_tvp_fun

        self._mpc.set_tvp_fun(mpc_tvp_fun)

        # Setup
        self._mpc.setup()
        self._initial_guess_set = False
        # # set initial guess
        # self._mpc.x0 = np.array(init_state)
        # self._mpc.set_initial_guess()
        # # set basic goal
        # self.set_new_goal(init_state,
        #                   goal)

    def step(self, robot_state: np.ndarray, ped_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError()

    def predict(self,
                observation: Dict[int, np.ndarray]) -> np.ndarray:
        """Method predicts pedestrian trajectories with specified predictor

        Args:
            ground_truth_pedestrians_state (np.ndarray): Current state of the pedestrians, [2-d numpy array]
        """
        tracked_predictions = self._ped_tracker.get_predictions()
        predicted_trajectories = np.tile(np.array([10000., 10000., 0., 0.]), (self._horizon + 1, self._total_peds, 1))
        predicted_covs = np.tile(np.array([[0.01, 0.0], [0., 0.01]]), (self._horizon + 1, self._total_peds, 1, 1))
        for k in tracked_predictions.keys():
            predicted_trajectories[:, k, :2] = tracked_predictions[k][0]
            predicted_covs[:, k, :] = tracked_predictions[k][1]

        predicted_covs_inv = np.linalg.inv(predicted_covs)
        predicted_covs_flatten = predicted_covs.reshape((self._horizon + 1, self._total_peds, 4))
        predicted_covs_inv_flatten = predicted_covs_inv.reshape((self._horizon + 1, self._total_peds, 4))

        for step in range(len(self._mpc_tvp_fun['_tvp', :, 'p_peds'])):
            self._mpc_tvp_fun['_tvp', step, 'p_peds'] = predicted_trajectories[step].T
            self._mpc_tvp_fun['_tvp', step, 'cov_peds'] = predicted_covs_flatten[step].T
            self._mpc_tvp_fun['_tvp', step, 'inv_cov_peds'] = predicted_covs_inv_flatten[step].T
        return predicted_trajectories, predicted_covs

    def get_predicted_robot_trajectory(self) -> List[float]:
        rob_x_pred = self._mpc.data.prediction(('_x', 'x'))[0]
        rob_y_pred = self._mpc.data.prediction(('_x', 'y'))[0]
        array_xy_pred = np.concatenate([rob_x_pred, rob_y_pred], axis=1)
        return np.ndarray.tolist(array_xy_pred)

    def make_step(self,
                  state: np.ndarray,
                  observation: Dict[int, np.ndarray]) -> np.ndarray:
        self._predicted_pedestrians_trajectories, self._predicted_pedestrians_covariances = self.predict(observation)
        control = self._mpc.make_step(state).T[0]
        return control, self._predicted_pedestrians_trajectories, self._predicted_pedestrians_covariances

    def set_new_goal(self,
                     current_state: np.ndarray,
                     new_goal: np.ndarray) -> None:
        super().set_new_goal(current_state,
                             new_goal)
        self._mpc_tvp_fun['_tvp', :, 'current_goal'] = self.goal

    @property
    def predictor(self) -> AbstractPredictor:
        return self._predictor

    @property
    def init_state(self) -> np.ndarray:
        return self._init_state

    @property
    def ghost_tracking_times(self) -> List[int]:
        return self._ghost_tracking_times

    @property
    def max_ghost_tracking_time(self) -> int:
        return self._max_ghost_tracking_time

    @property
    def predicted_pedestrians_trajectories(self) -> np.ndarray:
        return self._predicted_pedestrians_trajectories


    @staticmethod
    def _determinant(matrix):
        det = (matrix[0, 0] * matrix[1, 1]) - (matrix[1, 0] * matrix[0, 1])
        return det


# class AbstractController(ABC):
#
#     def __init__(self,
#                  init_state: np.ndarray,
#                  goal: np.ndarray,
#                  horizon: int,
#                  dt: float,
#                  state_dummy_ped: List[float],
#                  max_ghost_tracking_time: int) -> None:
#         assert len(
#             goal) == 3, f"Goal should be provided with a vector [x, y, theta] and has a length of 3, provided length is {len(goal)}"
#         self._init_state = init_state
#         self._goal = goal
#         self._horizon = horizon
#         self._dt = dt
#         self._previously_detected_pedestrians: Set[int] = set()
#         self._max_ghost_tracking_time = max_ghost_tracking_time
#         self._state_dummy_ped = state_dummy_ped
#
#     @abstractmethod
#     def make_step(self,
#                   state: np.ndarray,
#                   ground_truth_pedestrians_state: np.ndarray) -> np.ndarray:
#         """Method calculates control input for the specified system
#         Args:
#             state (np.ndarray): State of the system
#             propagation_ped_traj (np.ndarray): Predicted pedestrian trajectories [prediction step, pedestrian, [x, y, vx, vy]]
#
#         Raises:
#             NotImplementedError: Abstract method was not implemented
#         Returns:
#             np.ndarray: Control input
#         """
#         raise NotImplementedError()
#
#     @property
#     def init_state(self) -> np.ndarray:
#         return self._init_state
#
#     @property
#     def goal(self) -> np.ndarray:
#         return self._goal
#
#     @goal.setter
#     def goal(self,
#              new_goal: np.ndarray) -> np.ndarray:
#         self._goal = new_goal
#
#     @property
#     def horizon(self) -> np.ndarray:
#         return self._horizon
#
#     @property
#     def dt(self) -> float:
#         return self._dt
#
#     @property
#     def previously_detected_pedestrians(self) -> Set[int]:
#         return self._previously_detected_pedestrians
#
#     @previously_detected_pedestrians.setter
#     def previously_detected_pedestrians(self,
#                                         new_detected_pedestrians: Set[int]) -> Set[int]:
#         self._previously_detected_pedestrians = new_detected_pedestrians
#
#     def get_pedestrains_ghosts_states(self,
#                                       ground_truth_pedestrians_state: np.ndarray,
#                                       undetected_pedestrian_indices: List[int]) -> np.ndarray:
#         """Methods returns states for the pedestrians taking into account the ghosts.
#         Ghost is a pedestrian that was previously seen and dissepeared from the vision field.
#         This method helps the controller to take into account stored trajectories of the ghosts
#         Args:
#             ground_truth_pedestrians_state (np.ndarray): This is the structure of ground truth states. Here every ghost has a dummy pedestrian state.
#             undetected_pedestrian_indices (List[int]): Structure to keep undetected pedestrians at this time step.
#         Returns:
#             np.ndarray: Structure with ground truth states of detected pedestrians, memorized states of the missed but previously seen pedestrians and dummy states for previously not seen pedestrians.
#         """
#         pedestrains_ghosts_states = np.copy(ground_truth_pedestrians_state)
#         for undetected_pedestrian in undetected_pedestrian_indices:
#             if undetected_pedestrian in self._previously_detected_pedestrians:
#                 pedestrains_ghosts_states[undetected_pedestrian] = self._predicted_pedestrians_trajectories[
#                     1, undetected_pedestrian]
#                 self._ghost_tracking_times[undetected_pedestrian] += 1
#                 if self._ghost_tracking_times[undetected_pedestrian] > self._max_ghost_tracking_time:
#                     self._ghost_tracking_times[undetected_pedestrian] = 0
#                     pedestrains_ghosts_states[undetected_pedestrian] = self._state_dummy_ped
#                     self._previously_detected_pedestrians.remove(undetected_pedestrian)
#         return pedestrains_ghosts_states
#
#     def get_ref_direction(self,
#                           state: np.ndarray,
#                           goal: np.ndarray) -> np.ndarray:
#         """Get direction angle of reference position
#         This helps MPC to find solution when the reference point is located behind the robot.
#         Args:
#             state (np.ndarray): initial robot state vector (for more details see config file)
#             goal (np.ndarray): reference robot position (for more details see config file)
#         Returns:
#             np.ndarray: goal (np.ndarray): reference robot position with a changed direction
#         """
#         x_goal_polar = goal[0] - state[0]
#         y_goal_polar = goal[1] - state[1]
#         return np.array([goal[0], goal[1], np.arctan2(y_goal_polar, x_goal_polar)])
#
#     def set_new_goal(self,
#                      current_state: np.ndarray,
#                      new_goal: np.ndarray) -> None:
#         new_goal = self.get_ref_direction(current_state,
#                                           new_goal)
#         self.goal = new_goal
