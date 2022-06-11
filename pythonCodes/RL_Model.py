from keras.layers import Conv2D, AveragePooling2D, Activation, Flatten, Dense, Concatenate, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications import Xception
from keras.models import Sequential, Input, Model
from keras.callbacks import TensorBoard

from collections import deque

import tensorflow as tf
import random
import time
import numpy as np

import carla
import cv2
import math

IM_WIDTH = 640
IM_HEIGHT = 480
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5

ACTION_NUMBER = 7

DISCOUNT = 0.99

SECONDS_PER_EPISODE = 10

REPLAY_MEMORY_SIZE = 5_000
MODEL_NAME = "CNN_SEG"
SHOW_CAM = False


class ModifiedTensorBoard(TensorBoard):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)

    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0
        self.graph = tf.compat.v1.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):

        base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 3))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        predictions = Dense(ACTION_NUMBER, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        model.summary()
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch]) / 255
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255

        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        Y = []

        for index, (current_states, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_states)
            Y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        with self.graph.as_default():
            self.model.fit(np.array(X) / 255, np.array(Y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False,
                           callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, ACTION_NUMBER)).astype(np.float32)

        with self.graph.as_default():
            self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)


class CarEnv:
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    segmentation_camera = None

    def __init__(self, result=False):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.result = result

        self.vehicle = None

        self.handle_value = 0.0
        self.throttle_value = 0.0

        self.episode_start = 0

        # self.spawn_points = self.world.get_map().get_spawn_points()
        self.spawn_points = []  # world.get_map().get_spawn_points()

        self.actor_start_point = carla.Transform(carla.Location(-103, 0, 0.6), carla.Rotation(0, -90, 0))  # actor spawn point

        self.spawn_points.append(carla.Transform(carla.Location(-106.5, -10, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(-90.2, -50, 0.6), carla.Rotation(0, -45, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(-105.8, -30, 0.6), carla.Rotation(0, -80, 0)))

        self.spawn_points.append(carla.Transform(carla.Location(-30, -57.5, 0.6), carla.Rotation(0, 0, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(30, -57.5, 0.6), carla.Rotation(0, 0, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(50, -57.5, 0.6), carla.Rotation(0, 0, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(-20, -60.5, 0.6), carla.Rotation(0, 0, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(0, -60.5, 0.6), carla.Rotation(0, 0, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(20, -60.5, 0.6), carla.Rotation(0, 0, 0)))

        self.spawn_points.append(carla.Transform(carla.Location(-110.1, -28, 0.6), carla.Rotation(0, 90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(-113.5, -23, 0.6), carla.Rotation(0, 90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(-106.5, -40, 0.6), carla.Rotation(0, 115, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(-90.5, -63.5, 0.6), carla.Rotation(0, 150, 0)))

        self.spawn_points.append(carla.Transform(carla.Location(-20, -68.3, 0.6), carla.Rotation(0, 180, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(0, -68.3, 0.6), carla.Rotation(0, 180, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(40, -68.3, 0.6), carla.Rotation(0, 180, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(-10, -64.7, 0.6), carla.Rotation(0, 180, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(30, -64.7, 0.6), carla.Rotation(0, 180, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(60, -64.7, 0.6), carla.Rotation(0, 180, 0)))

        self.spawn_points.append(carla.Transform(carla.Location(-42, -30, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(-42, -10, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(-42, 5, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(-45.5, -25, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(-45.5, -5, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(-45.5, 10, 0.6), carla.Rotation(0, -90, 0)))

        self.spawn_points.append(carla.Transform(carla.Location(106.5, -17, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(106.5, -5, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(106.5, 0, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(106.5, 10, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(110, -15, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(110, -5, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(110, 10, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(110, 5, 0.6), carla.Rotation(0, -90, 0)))

        self.spawn_points.append(carla.Transform(carla.Location(106.5, 45, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(106.5, 55, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(106.5, 65, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(106.5, 75, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(110, 45, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(110, 55, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(110, 65, 0.6), carla.Rotation(0, -90, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(110, 75, 0.6), carla.Rotation(0, -90, 0)))

        self.world.get_spectator().set_transform(self.actor_start_point)
        self.vehicle_blueprints = self.blueprint_library.filter('*vehicle*')

    def reset(self):
        self.collision_hist = []
        self.invasion_hist = []
        self.actor_list = []
        self.vehicle_list = []

        while True:
            self.vehicle = self.world.try_spawn_actor(self.model_3, self.actor_start_point)
            if self.vehicle is not None:
                self.actor_list.append(self.vehicle)
                break

        for i in range(0, len(self.spawn_points)):
            tmp_vehicle = self.world.try_spawn_actor(random.choice(self.vehicle_blueprints), self.spawn_points[i])
            if tmp_vehicle is not None:
                tmp_vehicle.set_autopilot(True)
                self.vehicle_list.append(tmp_vehicle)

        seg_cam = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        seg_cam.set_attribute("image_size_x", f"{IM_WIDTH}")
        seg_cam.set_attribute("image_size_y", f"{IM_HEIGHT}")
        seg_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        seg_sensor = self.world.spawn_actor(seg_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(seg_sensor)
        seg_sensor.listen(lambda data: self.process_segmentation_img(data))

        time.sleep(4)

        col_sensor_bp = self.blueprint_library.find("sensor.other.collision")
        col_sensor = self.world.spawn_actor(col_sensor_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(col_sensor)
        col_sensor.listen(lambda event: self.collision_data(event))

        line_sensor_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        line_sensor = self.world.spawn_actor(line_sensor_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(line_sensor)
        line_sensor.listen(lambda event: self.on_invasion(event))

        while self.segmentation_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

        return self.segmentation_camera

    def get_state(self):
        return self.segmentation_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def on_invasion(self, event):
        self.invasion_hist.append(event)

    def process_segmentation_img(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        if SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.segmentation_camera = i3

    def step(self, action):
        if action == 0:
            if self.throttle_value <= 0.95:
                self.throttle_value += 0.05
        elif action == 1:
            if self.throttle_value >= 0.05:
                self.throttle_value -= 0.05
        elif action == 2:
            if self.handle_value >= -0.95:
                self.handle_value -= 0.05
        elif action == 3:
            if self.handle_value <= 0.95:
                self.handle_value += 0.05
        elif action == 4:
            pass

        # print(f'throttle: {self.throttle_value} steer" {self.handle_value}')
        self.vehicle.apply_control(carla.VehicleControl(throttle=self.throttle_value, steer=self.handle_value))

        if self.result:
            return self.segmentation_camera, 0, False, None

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        reward = 0

        # 충돌 reward
        if len(self.collision_hist) != 0:
            done = True
            reward -= 500
        else:
            waypoint = self.client.get_world().get_map().get_waypoint(self.vehicle.get_location(), project_to_road=True)

            vehicle_location = self.vehicle.get_location()
            waypoint_location = waypoint.transform.location
            distance = math.sqrt(math.pow((vehicle_location.x - waypoint_location.x), 2)
                                 + math.pow((vehicle_location.y - waypoint_location.y), 2)
                                 + math.pow((vehicle_location.z - waypoint_location.z), 2))

            vehicle_forward = self.vehicle.get_transform().get_forward_vector()
            waypoint_forward = waypoint.transform.get_forward_vector()
            deltaAngle = math.acos(
                (
                            vehicle_forward.x * waypoint_forward.x + vehicle_forward.y * waypoint_forward.y + vehicle_forward.z * waypoint_forward.z) /
                math.sqrt(
                    vehicle_forward.x * vehicle_forward.x + vehicle_forward.y * vehicle_forward.y + vehicle_forward.z * vehicle_forward.z) /
                math.sqrt(
                    waypoint_forward.x * waypoint_forward.x + waypoint_forward.y * waypoint_forward.y + waypoint_forward.z * waypoint_forward.z)
            )
            deltaAngle *= math.degrees(deltaAngle)

            # 도로 중심으로 부터 거리 & 도로 방향과 차이 각도 reward
            distance = int(distance * 10)
            deltaAngle = int(deltaAngle)

            reward -= distance
            reward -= deltaAngle

            # 차량 속도에 따른 reward
            velocity_value = int(kmh * kmh / 100 - 1)
            reward +=velocity_value

            # 이동 거리에 따른 reward
            mileage = math.sqrt(math.pow((vehicle_location.x - self.actor_start_point.location.x), 2)
                                 + math.pow((vehicle_location.y - self.actor_start_point.location.y), 2)
                                 + math.pow((vehicle_location.z - self.actor_start_point.location.z), 2))
            mileage = int(mileage * 10)
            reward += mileage

            # print(f'distance -{distance} | deltaAngle -{deltaAngle} | velocity {velocity_value} | mileage {mileage}')

            done = False

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.segmentation_camera, reward, done, None

    def destroy_actors(self):
        print(f'destroy actors({len(self.actor_list)})')
        # self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicle_list])

        for actor in self.actor_list:
                actor.destroy()

        self.actor_list.clear()
        self.vehicle_list.clear()
