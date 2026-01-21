#!/usr/bin/env python3

# Copyright 2023 Clearpath Robotics, Inc.
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
#
# @author Hilary Luo (hluo@clearpathrobotics.com)

from math import floor
from threading import Lock, Thread
from time import sleep
import signal
import sys

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import BatteryState
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator

BATTERY_HIGH = 0.95
BATTERY_LOW = 0.2
BATTERY_CRITICAL = 0.1

class BatteryMonitor(Node):
    def __init__(self, lock):
        super().__init__('battery_monitor')
        self.lock = lock
        self.battery_percent = None  # Initialize to None

        # Subscribe to the /battery_state topic
        self.battery_state_subscriber = self.create_subscription(
            BatteryState,
            'battery_state',
            self.battery_state_callback,
            qos_profile_sensor_data)

    def battery_state_callback(self, batt_msg: BatteryState):
        with self.lock:
            self.battery_percent = batt_msg.percentage

    def thread_function(self):
        executor = SingleThreadedExecutor()
        executor.add_node(self)
        try:
            executor.spin()
        except rclpy.exceptions.ROSInterruptException:
            pass
        finally:
            executor.shutdown()

def signal_handler(sig, frame):
    print("Shutting down gracefully...")
    rclpy.shutdown()
    sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    signal.signal(signal.SIGINT, signal_handler)

    lock = Lock()
    battery_monitor = BatteryMonitor(lock)
    navigator = TurtleBot4Navigator()

    thread = Thread(target=battery_monitor.thread_function, daemon=True)
    thread.start()

    try:
        # Start on dock
        if not navigator.getDockedStatus():
            navigator.info('Docking before initializing pose')
            navigator.dock()

        # Set initial pose
        initial_pose = navigator.getPoseStamped([0.0, 0.0], TurtleBot4Directions.NORTH)
        navigator.setInitialPose(initial_pose)

        # Wait for Nav2
        navigator.waitUntilNav2Active()

        # Undock
        navigator.undock()

        # Prepare goal poses
        goal_poses = [
            navigator.getPoseStamped([2.624, 3.326], TurtleBot4Directions.EAST), 
            navigator.getPoseStamped([1.214, 8.903], TurtleBot4Directions.EAST),
            navigator.getPoseStamped([-2.238, 8.815], TurtleBot4Directions.NORTH),
            navigator.getPoseStamped([-2.172, 1.532], TurtleBot4Directions.NORTH_WEST),
            navigator.getPoseStamped([-1.388, -3.703], TurtleBot4Directions.WEST)
        ]

        position_index = 0
        while rclpy.ok():
            with lock:
                battery_percent = battery_monitor.battery_percent

            if battery_percent is None:
                navigator.info('Waiting for battery state...')
                sleep(1)
                continue

            navigator.info(f'Battery is at {(battery_percent*100):.2f}% charge')

            if battery_percent < BATTERY_CRITICAL:
                navigator.error('Battery critically low. Shutting down.')
                break
            elif battery_percent < BATTERY_LOW:
                navigator.info('Navigating to docking area')
                navigator.startToPose(navigator.getPoseStamped([-1.0, 1.0], TurtleBot4Directions.EAST))
                
                navigator.info('Docking for charge')
                for attempt in range(3):  # Retry docking up to 3 times
                    navigator.dock()
                    if navigator.getDockedStatus():
                        break
                    navigator.error(f'Docking attempt {attempt + 1} failed')
                    sleep(5)
                else:
                    navigator.error('Robot failed to dock after retries')
                    break

                # Wait until charged
                navigator.info('Charging...')
                while rclpy.ok() and battery_percent < BATTERY_HIGH:
                    sleep(10)  # Reduced sleep for responsiveness
                    with lock:
                        battery_percent = battery_monitor.battery_percent
                        if battery_percent is None:
                            continue
                        battery_percent_prev = floor(battery_percent * 100) / 100
                    if battery_percent > battery_percent_prev + 0.01:
                        navigator.info(f'Battery is at {(battery_percent*100):.2f}% charge')

                # Undock
                navigator.undock()
                position_index = 0
            else:
                # Navigate to next position
                navigator.info(f'Navigating to pose {position_index + 1}')
                navigator.startToPose(goal_poses[position_index])
                position_index = (position_index + 1) % len(goal_poses)

    except KeyboardInterrupt:
        navigator.info('Interrupted by user')
    finally:
        battery_monitor.destroy_node()
        rclpy.shutdown()
        thread.join()

if __name__ == '__main__':
    main()