# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Episodic rollout strategy: mirrors the behavior of ``lerobot-record``.

- Policy drives the robot during each recording episode.
- An optional teleoperator can drive the robot during reset phases so the
  operator can bring the environment back to its starting configuration.
  If no teleop is connected the robot stays in its current position.
- Keyboard controls:

      Right arrow  — end the current episode or reset phase early
      Left arrow   — discard the current episode and re-record it
      Escape       — stop the recording session

Dataset naming follows the rollout convention: repo names must start with ``rollout_``.
"""

from __future__ import annotations

import contextlib
import logging
import time

from lerobot.common.control_utils import (
    follower_smooth_move_to,
    teleop_smooth_move_to,
    teleop_supports_feedback,
)
from lerobot.datasets import VideoEncodingManager
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import log_visualization_data

from ..configs import EpisodicStrategyConfig
from ..context import RolloutContext
from .core import RolloutStrategy, safe_push_to_hub, send_next_action

logger = logging.getLogger(__name__)


class EpisodicStrategy(RolloutStrategy):
    """Policy-driven multi-episode recording, mirrors the behavior of ``lerobot-record``.

    Each recording episode runs the policy for maximum ``dataset.episode_time_s``
    seconds, recording one frame per policy action (``1/fps`` cadence — with
    ``interpolation_multiplier > 1`` the interpolated ticks only send commands
    to the robot).  A reset phase of ``dataset.reset_time_s``
    follows every episode (except the last) so the operator can manually
    reset the environment.  During the reset phase, an optional teleoperator
    drives the robot; if none is present the robot returns to its initial joint positions captured at startup.

    The policy state (hidden state, RTC queue, interpolator) is reset at
    the start of each recording episode.

    Keyboard events:
        right arrow  → end current episode or reset phase early
        left arrow   → discard & re-record current episode
        ESC          → stop the session
    """

    config: EpisodicStrategyConfig

    def __init__(self, config: EpisodicStrategyConfig) -> None:
        super().__init__(config)
        self._listener = None
        self._events: dict | None = None

    def setup(self, ctx: RolloutContext) -> None:
        """Start the inference engine and attach the keyboard listener."""
        self._init_engine(ctx)
        self._listener, self._events = init_keyboard_listener()
        logger.info("Episodic strategy ready")

    def run(self, ctx: RolloutContext) -> None:
        """Main multi-episode recording loop."""
        cfg = ctx.runtime.cfg
        dataset_cfg = cfg.dataset
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        dataset = ctx.data.dataset
        events = self._events
        features = ctx.data.dataset_features

        fps = cfg.fps
        episode_time_s = dataset_cfg.episode_time_s
        reset_time_s = dataset_cfg.reset_time_s
        num_episodes = dataset_cfg.num_episodes
        single_task = dataset_cfg.single_task or cfg.task
        play_sounds = cfg.play_sounds

        display_compressed = (
            True
            if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
            else cfg.display_compressed_images
        )

        # One timer for the whole session: episodes get their own cadence line, and
        # the run summary averages across them without the untimed reset phases.
        timer = CycleTimer(fps, self._interpolator.multiplier)

        encoding_manager = (
            VideoEncodingManager(dataset) if self.config.record_episodes else contextlib.nullcontext()
        )
        with encoding_manager:
            try:
                recorded_episodes = 0
                while recorded_episodes < num_episodes and not events["stop_recording"]:
                    if ctx.runtime.shutdown_event.is_set():
                        break

                    # Reset policy state at episode start (discard leftover hidden state / queue)
                    self._engine.reset()
                    self._interpolator.reset()
                    # A reset interpolator re-primes over two consecutive inference
                    # ticks, exactly like loop start-up, so exempt the group that
                    # spans them instead of reporting a healthy episode as slow.
                    timer.restart()
                    self._engine.resume()

                    self._refresh_grasp_target(ctx)
                    episode_index = dataset.num_episodes if self.config.record_episodes else recorded_episodes
                    verb = "Recording" if self.config.record_episodes else "Running"
                    log_say(f"{verb} episode {episode_index}", play_sounds)
                    self._policy_loop(
                        ctx=ctx,
                        robot=robot,
                        events=events,
                        features=features,
                        timer=timer,
                        control_time_s=episode_time_s,
                        dataset=dataset,
                        single_task=single_task,
                    )

                    # Reset phase, skip after the last episode (but run when re-recording)
                    if not events["stop_recording"] and (
                        recorded_episodes < num_episodes - 1 or events["rerecord_episode"]
                    ):
                        log_say("Reset the environment", play_sounds)

                        if teleop:
                            # Smooth handover so the transition to teleop control is jerk-free.
                            # For actuated teleops: drive the leader arm to the follower's current
                            # position so the operator takes over without fighting the arm.
                            # For non-actuated teleops: slide the follower to the teleop's current
                            # pose instead, since the leader cannot be driven.
                            # Disabled entirely with --strategy.smooth_handover=false (useful for
                            # clutch-style teleops that re-reference at the current robot pose on
                            # engage).
                            if self.config.smooth_handover:
                                obs = robot.get_observation()
                                current_pos = {k: v for k, v in obs.items() if k.endswith(".pos")}
                                if (
                                    teleop_supports_feedback(teleop)
                                    and self.config.smooth_leader_to_follower_handover
                                ):
                                    logger.info("Smooth handover: moving leader arm to follower position")
                                    teleop_smooth_move_to(teleop, current_pos, duration_s=2)
                                    teleop.disable_torque()
                                else:
                                    logger.info("Smooth handover: sliding follower to teleop position")
                                    teleop_action = teleop.get_action()
                                    processed = ctx.processors.teleop_action_processor((teleop_action, obs))
                                    target = ctx.processors.robot_action_processor((processed, obs))
                                    follower_smooth_move_to(robot, current_pos, target, duration_s=1)

                        elif self.config.reset_to_initial_position:
                            # No teleop: return the robot to its startup position.
                            self.return_to_initial_position(hw=ctx.hardware, duration_s=1)

                        self._reset_loop(
                            ctx=ctx,
                            robot=robot,
                            teleop=teleop,
                            events=events,
                            fps=fps,
                            control_time_s=reset_time_s,
                            display_data=cfg.display_data,
                            display_mode=cfg.display_mode,
                            display_compressed=display_compressed,
                        )

                    if events["rerecord_episode"]:
                        log_say("Re-record episode", play_sounds)
                        events["rerecord_episode"] = False
                        events["exit_early"] = False
                        if self.config.record_episodes:
                            dataset.clear_episode_buffer()
                        timer.log_episode_summary("discarded episode")

                        # returns to its initial joint positions captured at startup
                        if not teleop and self.config.reset_to_initial_position:
                            self.return_to_initial_position(hw=ctx.hardware, duration_s=1)

                        continue

                    if self.config.record_episodes:
                        dataset.save_episode()
                    recorded_episodes += 1
                    timer.log_episode_summary(f"episode {recorded_episodes}")
            finally:
                # Save any frames buffered in the current episode so an unexpected
                # exception or KeyboardInterrupt does not silently drop recorded data.
                # suppress: save_episode raises if the buffer is empty (nothing to lose).
                logger.info("Episodic control loop ended")
                timer.log_run_summary()
                if self.config.record_episodes:
                    logger.info("Saving any in-progress episode")
                    with contextlib.suppress(Exception):
                        dataset.save_episode()

    def _policy_loop(
        self,
        ctx: RolloutContext,
        robot,
        events: dict,
        features: dict,
        timer: CycleTimer,
        control_time_s: float,
        dataset,
        single_task: str,
    ) -> None:
        """Policy-driven recording loop for a single episode.

        *timer* is owned by :meth:`run` and shared across episodes so its run
        summary spans the session; the caller re-arms it between episodes.
        """
        interpolator = self._interpolator

        timestamp = 0.0
        start_t = time.perf_counter()
        auto_next_departed = False
        auto_next_stable_since_t: float | None = None
        auto_next_stable_position: dict[str, float] | None = None
        auto_next_hold_s = self.config.auto_next_motion_hold_s
        auto_next_was_in_end_region = False
        auto_next_last_log_t = 0.0
        auto_next_last_stable_log = -1.0
        initial_position = ctx.hardware.initial_position

        if self.config.auto_next_on_settle:
            logger.info(
                "Auto-next enabled: leave>=%.1f end_tol=%.1f motion<=%.1f hold=%.2fs "
                "min_episode=%.1fs log_every=%.1fs end_pose=%s",
                self.config.auto_next_leave_tolerance,
                self.config.auto_next_end_tolerance,
                self.config.auto_next_motion_tolerance,
                auto_next_hold_s,
                self.config.auto_next_min_episode_s,
                self.config.auto_next_log_interval_s,
                self.config.auto_next_end_position,
            )

        while timestamp < control_time_s:
            timer.tick(new_cycle=interpolator.needs_new_action())

            if events["exit_early"]:
                events["exit_early"] = False
                break

            if ctx.runtime.shutdown_event.is_set():
                break

            with timer.section("observe"):
                obs = robot.get_observation()

            if self.config.auto_next_on_settle and initial_position:
                elapsed = time.perf_counter() - start_t
                current_position = {key: float(obs[key]) for key in initial_position if key in obs}
                end_position = self.config.auto_next_end_position
                if end_position is not None and len(end_position) != len(current_position):
                    raise ValueError(
                        "auto_next_end_position must contain one value per robot position feature "
                        f"({len(current_position)} expected, got {len(end_position)})"
                    )
                end_distance = (
                    sum(
                        abs(value - target)
                        for value, target in zip(current_position.values(), end_position, strict=True)
                    )
                    if end_position is not None
                    else 0.0
                )
                in_end_region = end_position is None or end_distance <= self.config.auto_next_end_tolerance
                leave_distance = sum(
                    abs(current_position[key] - float(value))
                    for key, value in initial_position.items()
                    if key in current_position
                )
                if not auto_next_departed and leave_distance >= self.config.auto_next_leave_tolerance:
                    auto_next_departed = True
                    auto_next_stable_since_t = None
                    auto_next_stable_position = current_position
                    logger.info(
                        "Auto-next armed: robot left startup pose (leave=%.2f >= %.2f)",
                        leave_distance,
                        self.config.auto_next_leave_tolerance,
                    )

                motion = 0.0
                stable_elapsed = 0.0
                if auto_next_departed and auto_next_stable_position is not None:
                    if in_end_region and not auto_next_was_in_end_region:
                        logger.info(
                            "Auto-next: robot entered end-pose region (end=%.2f <= %.2f); "
                            "waiting for motion<=%.1f for %.2fs",
                            end_distance,
                            self.config.auto_next_end_tolerance,
                            self.config.auto_next_motion_tolerance,
                            auto_next_hold_s,
                        )
                    auto_next_was_in_end_region = in_end_region

                    motion = sum(
                        abs(current_position[key] - value)
                        for key, value in auto_next_stable_position.items()
                    )
                    now_t = time.perf_counter()
                    if not in_end_region or motion > self.config.auto_next_motion_tolerance:
                        if auto_next_stable_since_t is not None:
                            reason = (
                                f"left end region (end={end_distance:.2f})"
                                if not in_end_region
                                else (
                                    f"motion={motion:.2f} > "
                                    f"{self.config.auto_next_motion_tolerance:.1f}"
                                )
                            )
                            logger.info(
                                "Auto-next stable RESET: %s (had %.2f/%.2fs) leave=%.2f end=%.2f",
                                reason,
                                now_t - auto_next_stable_since_t,
                                auto_next_hold_s,
                                leave_distance,
                                end_distance,
                            )
                        auto_next_stable_since_t = None
                        auto_next_last_stable_log = -1.0
                        auto_next_stable_position = current_position
                    else:
                        if auto_next_stable_since_t is None:
                            auto_next_stable_since_t = now_t
                            logger.info(
                                "Auto-next stable START: motion=%.2f <= %.1f; "
                                "need %.2fs continuous (leave=%.2f end=%.2f)",
                                motion,
                                self.config.auto_next_motion_tolerance,
                                auto_next_hold_s,
                                leave_distance,
                                end_distance,
                            )
                        stable_elapsed = now_t - auto_next_stable_since_t
                        # Log progress every ~0.2s while accumulating in the end region.
                        if stable_elapsed - auto_next_last_stable_log >= 0.2:
                            auto_next_last_stable_log = stable_elapsed
                            logger.info(
                                "Auto-next stable: %.2f/%.2fs motion=%.2f leave=%.2f end=%.2f",
                                stable_elapsed,
                                auto_next_hold_s,
                                motion,
                                leave_distance,
                                end_distance,
                            )
                        if (
                            elapsed >= self.config.auto_next_min_episode_s
                            and stable_elapsed >= auto_next_hold_s
                        ):
                            logger.info(
                                "Auto-next TRIGGER: enter next episode "
                                "(leave=%.2f end=%.2f motion=%.2f stable=%.2f/%.2fs elapsed=%.1fs)",
                                leave_distance,
                                end_distance,
                                motion,
                                stable_elapsed,
                                auto_next_hold_s,
                                elapsed,
                            )
                            precise_sleep(self.config.auto_next_settle_s)
                            break

                if (
                    self.config.auto_next_log_interval_s > 0
                    and time.perf_counter() - auto_next_last_log_t >= self.config.auto_next_log_interval_s
                ):
                    auto_next_last_log_t = time.perf_counter()
                    logger.info(
                        "Auto-next check: elapsed=%.1fs leave=%.2f/%s end=%.2f/%s motion=%.2f/%s "
                        "stable=%.2f/%.2fs departed=%s in_end=%s ready=%s",
                        elapsed,
                        leave_distance,
                        f">={self.config.auto_next_leave_tolerance:.1f}",
                        end_distance,
                        (
                            f"<={self.config.auto_next_end_tolerance:.1f}"
                            if end_position is not None
                            else "n/a"
                        ),
                        motion,
                        f"<={self.config.auto_next_motion_tolerance:.1f}",
                        stable_elapsed,
                        auto_next_hold_s,
                        auto_next_departed,
                        in_end_region,
                        (
                            auto_next_departed
                            and in_end_region
                            and elapsed >= self.config.auto_next_min_episode_s
                            and stable_elapsed >= auto_next_hold_s
                        ),
                    )

            with timer.section("process_obs"):
                obs_processed = self._process_observation_and_notify(ctx, obs)

            if self._handle_warmup(ctx.runtime.cfg.use_torch_compile, timer):
                continue

            action_dict = send_next_action(obs_processed, obs, ctx, interpolator, timer)

            if action_dict is not None:
                with timer.section("telemetry"):
                    self._log_telemetry(obs_processed, action_dict, ctx.runtime, ctx.grasp_target)
                # Record once per interpolation cycle so the dataset cadence
                # matches its declared fps; interpolated ticks only send
                # commands to the robot.
                if self.config.record_episodes and interpolator.emitted_policy_action:
                    with timer.section("record"):
                        obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                        action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                        dataset.add_frame({**obs_frame, **action_frame, "task": single_task})

            timer.wait()
            timestamp = time.perf_counter() - start_t

    def _reset_loop(
        self,
        ctx: RolloutContext,
        robot,
        teleop,
        events: dict,
        fps: float,
        control_time_s: float,
        display_data: bool,
        display_mode: str,
        display_compressed: bool,
    ) -> None:
        """Reset-phase loop: teleop drives the robot if available, no recording."""
        processors = ctx.processors
        control_interval = 1.0 / fps

        timestamp = 0.0
        start_t = time.perf_counter()

        while timestamp < control_time_s:
            loop_start = time.perf_counter()

            if events["exit_early"]:
                events["exit_early"] = False
                break

            if ctx.runtime.shutdown_event.is_set():
                break

            obs = robot.get_observation()

            if teleop is not None:
                act = teleop.get_action()
                act_teleop = processors.teleop_action_processor((act, obs))
                robot_action = processors.robot_action_processor((act_teleop, obs))
                robot.send_action(robot_action)

                if display_data:
                    obs_processed = processors.robot_observation_processor(obs)
                    log_visualization_data(
                        display_mode,
                        observation=obs_processed,
                        action=act_teleop,
                        compress_images=display_compressed,
                    )

            dt = time.perf_counter() - loop_start
            sleep_t = control_interval - dt
            precise_sleep(max(sleep_t, 0.0))
            timestamp = time.perf_counter() - start_t

    def teardown(self, ctx: RolloutContext) -> None:
        """Finalise dataset, stop listener, push to hub, and disconnect hardware."""
        cfg = ctx.runtime.cfg
        play_sounds = cfg.play_sounds

        log_say("Stop recording", play_sounds, blocking=True)

        if self._listener is not None:
            self._listener.stop()

        if self.config.record_episodes and ctx.data.dataset is not None:
            logger.info("Finalizing dataset...")
            ctx.data.dataset.finalize()

        if (
            self.config.record_episodes
            and cfg.dataset is not None
            and cfg.dataset.push_to_hub
            and ctx.data.dataset is not None
            and safe_push_to_hub(
                ctx.data.dataset,
                tags=cfg.dataset.tags,
                private=cfg.dataset.private,
            )
        ):
            logger.info("Dataset uploaded to hub")
            log_say("Dataset uploaded to hub", play_sounds)

        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=cfg.return_to_initial_position,
        )
        log_say("Exiting", play_sounds)
        logger.info("Episodic strategy teardown complete")
