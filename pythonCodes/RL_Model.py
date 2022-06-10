from keras.layers import Conv2D, AveragePooling2D, Activation, Flatten, Dense, Concatenate
from keras.optimizers import Adam
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

ACTION_NUMBER = 6

DISCOUNT = 0.99

SECONDS_PER_EPISODE = 10

REPLAY_MEMORY_SIZE = 5_000
MODEL_NAME = "CNN_SEG"
SHOW_CAM = True


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
        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, strides=3, input_shape=(IM_HEIGHT, IM_WIDTH, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, kernel_size=3, strides=3, padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, kernel_size=3, strides=3, padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Flatten())
        model.add(Dense(ACTION_NUMBER, input_dim=64, activation='relu'))

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
    STEER_AMT = 1.0
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
        self.vehicle_list = []
        self.walker_list = []
        self.walker_controller_list = []

        self.brake_value = 0.0
        self.throttle_value = 1.0
        self.handle_value = 0.0

        self.episode_start = 0

        self.spawn_points = self.world.get_map().get_spawn_points()
        vehicle_blueprints = self.blueprint_library.filter('*vehicle*')
        for i in range(1, len(self.spawn_points), 3):
            tmp_vehicle = self.world.try_spawn_actor(random.choice(vehicle_blueprints), self.spawn_points[i])
            if tmp_vehicle is not None:
                tmp_vehicle.set_autopilot(True)
                self.vehicle_list.append(tmp_vehicle)

        walker_blueprints = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        walker_control_blueprint = self.blueprint_library.find('controller.ai.walker')
        for i in range(50):
            spawn_point = carla.Transform()
            spawn_point.location = self.world.get_random_location_from_navigation()
            tmp_walker = self.world.try_spawn_actor(random.choice(walker_blueprints), spawn_point)

            if tmp_walker is not None:
                tmp_controller = self.world.try_spawn_actor(walker_control_blueprint, carla.Transform(), attach_to=tmp_walker)
                self.walker_list.append(tmp_walker)
                self.walker_controller_list.append(tmp_controller)

        self.world.wait_for_tick()
        for controller in self.walker_controller_list:
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(1 + random.random())

    def reset(self):
        self.collision_hist = []
        self.invasion_hist = []
        self.actor_list = []

        transform = self.spawn_points[10]
        while True:
            self.vehicle = self.world.try_spawn_actor(self.model_3, transform)
            if self.vehicle is not None:
                self.actor_list.append(self.vehicle)
                break

        seg_cam = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        seg_cam.set_attribute("image_size_x", f"{IM_WIDTH}")
        seg_cam.set_attribute("image_size_y", f"{IM_HEIGHT}")
        seg_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        seg_sensor = self.world.spawn_actor(seg_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(seg_sensor)
        seg_sensor.listen(lambda data: self.process_segmentation_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
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
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

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
            if self.throttle_value <= 0.9:
                self.throttle_value += 0.1
        elif action == 1:
            if self.throttle_value >= 0.1:
                self.throttle_value -= 0.1
        elif action == 2:
            self.handle_value = 1
        elif action == 3:
            self.handle_value = -1
        elif action == 4:
            self.brake_value = 1
        elif action == 5:
            self.brake_value = 0

        self.vehicle.apply_control(carla.VehicleControl(throttle=self.throttle_value, steer=self.handle_value * self.STEER_AMT, brake=self.brake_value))

        if self.result:
            return self.segmentation_camera, 0, False, None

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        reward = 0

        if len(self.collision_hist) != 0:
            done = True
            reward -= 200
        else:
            for hist in self.invasion_hist:
                line_type = hist.crossed_lane_markings[0].type
                if line_type == "Solid":
                    reward -= 100
                if line_type == "SolidSolid":
                    reward -= 100

            if kmh < 50:
                done = False
                reward -= 1
            else:
                done = False
                reward += 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.segmentation_camera, reward, done, None

    def destroy_actors(self, bAll_Actors=False):
        print(f'destroy actors({len(self.actor_list)})')
        # self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        if bAll_Actors:
            for walcon in self.walker_controller_list:
                walcon.stop()
                walcon.destroy()

            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicle_list])

        else:
            for actor in self.actor_list:
                actor.destroy()


