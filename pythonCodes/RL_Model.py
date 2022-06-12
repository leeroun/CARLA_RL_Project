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
MIN_REPLAY_MEMORY_SIZE = 500
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5

MODEL_INPUTS = 5
ACTION_NUMBER = 5

DISCOUNT = 0.99

SECONDS_PER_EPISODE = 10

REPLAY_MEMORY_SIZE = 1_000
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
        in_image_left = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
        in_image_right = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
        in_image_front = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
        in_lane = Input(
            shape=(11,))  # left_dis (1) right_dis (1) delta angle (1) left lane type (4) right lane type (4)

        out_left1 = Conv2D(64, kernel_size=3, strides=3, input_shape=(IM_HEIGHT, IM_WIDTH, 3), padding='same')(
            in_image_left)
        out_right1 = Conv2D(64, kernel_size=3, strides=3, input_shape=(IM_HEIGHT, IM_WIDTH, 3), padding='same')(
            in_image_right)
        out_front1 = Conv2D(64, kernel_size=3, strides=3, input_shape=(IM_HEIGHT, IM_WIDTH, 3), padding='same')(
            in_image_front)
        aout_left1 = Activation('relu')(out_left1)
        aout_right1 = Activation('relu')(out_right1)
        aout_front1 = Activation('relu')(out_front1)
        pout_left1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(aout_left1)
        pout_right1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(aout_right1)
        pout_front1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(aout_front1)

        out_left2 = Conv2D(64, kernel_size=3, strides=3, padding='same')(pout_left1)
        out_right2 = Conv2D(64, kernel_size=3, strides=3, padding='same')(pout_right1)
        out_front2 = Conv2D(64, kernel_size=3, strides=3, padding='same')(pout_front1)
        aout_left2 = Activation('relu')(out_left2)
        aout_right2 = Activation('relu')(out_right2)
        aout_front2 = Activation('relu')(out_front2)
        pout_left2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(aout_left2)
        pout_right2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(aout_right2)
        pout_front2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(aout_front2)

        out_left3 = Conv2D(64, kernel_size=3, strides=3, padding='same')(pout_left2)
        out_right3 = Conv2D(64, kernel_size=3, strides=3, padding='same')(pout_right2)
        out_front3 = Conv2D(64, kernel_size=3, strides=3, padding='same')(pout_front2)
        aout_left3 = Activation('relu')(out_left3)
        aout_right3 = Activation('relu')(out_right3)
        aout_front3 = Activation('relu')(out_front3)
        pout_left3 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(aout_left3)
        pout_right3 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(aout_right3)
        pout_front3 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(aout_front3)

        fout_left = Flatten()(pout_left3)
        fout_right = Flatten()(pout_right3)
        fout_front = Flatten()(pout_front3)

        out_lane1 = Dense(8, activation='relu')(in_lane)
        out_lane2 = Dense(4, activation='relu')(out_lane1)
        out_lane3 = Dense(1, activation='relu')(out_lane2)

        out = Concatenate(axis=1)([fout_left, fout_right, fout_front, out_lane3])
        dout = Dense(ACTION_NUMBER, activation='relu')(out)

        model = Model(inputs=[in_image_left, in_image_right, in_image_front, in_lane], outputs=dout)

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        model.summary()
        print(model.input_shape)
        return model

    def update_replay_memory(self, transition):
        transition = np.array(transition)
        current_transition = transition[0]
        new_transition = transition[3]

        lane_list = [current_transition[3][0], current_transition[3][1], current_transition[3][2]]
        lane_list += current_transition[3][3]
        lane_list += current_transition[3][4]
        cur_left = np.array(current_transition[0]).reshape(-1, *current_transition[0].shape)
        cur_right = np.array(current_transition[1]).reshape(-1, *current_transition[1].shape)
        cur_front = np.array(current_transition[2]).reshape(-1, *current_transition[2].shape)
        cur_lane = np.array(lane_list)
        cur_lane = np.expand_dims(cur_lane, axis=0)

        transition[0] = [cur_left, cur_right, cur_front, cur_lane]

        lane_list = [new_transition[3][0], new_transition[3][1], new_transition[3][2]]
        lane_list += new_transition[3][3]
        lane_list += new_transition[3][4]
        new_left = np.array(new_transition[0]).reshape(-1, *new_transition[0].shape)
        new_right = np.array(new_transition[1]).reshape(-1, *new_transition[1].shape)
        new_front = np.array(new_transition[2]).reshape(-1, *new_transition[2].shape)
        new_lane = np.array(lane_list)
        new_lane = np.expand_dims(new_lane, axis=0)

        transition[3] = [new_left, new_right, new_front, new_lane]

        self.replay_memory.append(transition)
        print(f'replay memory length : {len(self.replay_memory)}')


    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = [transition[0] for transition in minibatch]

        current_qs_list = []
        with self.graph.as_default():
            for i in range(MINIBATCH_SIZE):
                current_qs_list.append(self.model.predict(current_states[i], PREDICTION_BATCH_SIZE)[0])

        new_current_states = [transition[3] for transition in minibatch]

        future_qs_list = []
        with self.graph.as_default():
            for i in range(MINIBATCH_SIZE):
                future_qs_list.append(self.target_model.predict(new_current_states[i], PREDICTION_BATCH_SIZE)[0])

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
            for i in range(MINIBATCH_SIZE):
                _Y = np.expand_dims(np.array(Y[i]), axis=0)
                self.model.fit(X[i], _Y, batch_size=1, verbose=0, shuffle=False)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        lane_list = [state[3][0], state[3][1], state[3][2]]
        lane_list += state[3][3]
        lane_list += state[3][4]
        # [self.left_seg_camera, self.right_seg_camera, self.front_seg_camera, [lane_left_dis, lane_right_dis, lane_delta_angle, lane_left_type, lane_right_type]]
        qs_left = np.array(state[0]).reshape(-1, *state[0].shape)
        qs_right = np.array(state[1]).reshape(-1, *state[1].shape)
        qs_front = np.array(state[2]).reshape(-1, *state[2].shape)
        qs_lane = np.array(lane_list)
        qs_lane = np.expand_dims(qs_lane, axis=0)

        return self.model.predict([qs_left, qs_right, qs_front, qs_lane])[0]

    def train_in_loop(self):
        X_image_f = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        X_image_l = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        X_image_r = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)

        X_left_dis = np.random.uniform(size=1).astype(np.float32)
        X_right_dis = np.random.uniform(size=1).astype(np.float32)
        X_left_type = np.eye(4)[np.random.randint(0, 4)].astype(np.float32)
        X_right_type = np.eye(4)[np.random.randint(0, 4)].astype(np.float32)
        X_delta_angle = np.random.uniform(size=1).astype(np.float32)

        # left_dis (1) right_dis (1) delta angle (1) left lane type (4) right lane type (4)
        X_lane = np.concatenate((X_left_dis, X_right_dis, X_delta_angle, X_left_type, X_right_type))
        X_lane = np.expand_dims(X_lane, axis=0)

        # X = np.array([X_image_f, X_image_l, X_image_r, X_lane])
        y = np.random.uniform(size=(1, ACTION_NUMBER)).astype(np.float32)

        with self.graph.as_default():
            self.model.fit([X_image_f, X_image_l, X_image_r, X_lane], y, verbose=False, batch_size=1)

        self.training_initialized = True
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)


class CarEnv:
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_seg_camera = None
    left_seg_camera = None
    right_seg_camera = None

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

        self.actor_start_point = carla.Transform(carla.Location(-30, -57.5, 0.6), carla.Rotation(0, 0, 0))  # actor spawn point
        self.spawn_points.append(carla.Transform(carla.Location(-10, -57.5, 0.6), carla.Rotation(0, 0, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(18, -57.5, 0.6), carla.Rotation(0, 0, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(45, -57.5, 0.6), carla.Rotation(0, 0, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(-20, -60.5, 0.6), carla.Rotation(0, 0, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(11, -60.5, 0.6), carla.Rotation(0, 0, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(29, -60.5, 0.6), carla.Rotation(0, 0, 0)))

        self.spawn_points.append(carla.Transform(carla.Location(-14, -68.3, 0.6), carla.Rotation(0, 180, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(0, -68.3, 0.6), carla.Rotation(0, 180, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(10, -68.3, 0.6), carla.Rotation(0, 180, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(29, -68.3, 0.6), carla.Rotation(0, 180, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(45, -68.3, 0.6), carla.Rotation(0, 180, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(55, -68.3, 0.6), carla.Rotation(0, 180, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(-10, -64.7, 0.6), carla.Rotation(0, 180, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(3, -64.7, 0.6), carla.Rotation(0, 180, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(21, -64.7, 0.6), carla.Rotation(0, 180, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(30, -64.7, 0.6), carla.Rotation(0, 180, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(38, -64.7, 0.6), carla.Rotation(0, 180, 0)))
        self.spawn_points.append(carla.Transform(carla.Location(48, -64.7, 0.6), carla.Rotation(0, 180, 0)))

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
        print('reset environment')

        self.collision_hist = []
        self.invasion_hist = []
        self.actor_list = []
        self.vehicle_list = []

        print('-spawn actor')
        while True:
            self.vehicle = self.world.try_spawn_actor(self.model_3, self.actor_start_point)
            if self.vehicle is not None:
                self.actor_list.append(self.vehicle)
                break

        print('-spawn vehicles')
        for i in range(0, len(self.spawn_points)):
            tmp_vehicle = self.world.try_spawn_actor(random.choice(self.vehicle_blueprints), self.spawn_points[i])
            if tmp_vehicle is not None:
                tmp_vehicle.set_autopilot(True)
                self.vehicle_list.append(tmp_vehicle)

        seg_cam = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        seg_cam.set_attribute("image_size_x", f"{IM_WIDTH}")
        seg_cam.set_attribute("image_size_y", f"{IM_HEIGHT}")
        seg_cam.set_attribute("fov", f"110")

        front_spawn_point = carla.Transform(carla.Location(x=2.5, z=0.4))
        left_spawn_point = carla.Transform(carla.Location(x=2.5, z=0.5), carla.Rotation(0, -45, 0))
        right_spawn_point = carla.Transform(carla.Location(x=2.5, z=0.6), carla.Rotation(0, 45, 0))

        front_seg_sensor = self.world.spawn_actor(seg_cam, front_spawn_point, attach_to=self.vehicle)
        self.actor_list.append(front_seg_sensor)
        front_seg_sensor.listen(lambda data: self.process_segmentation_img(data, 0))

        left_seg_sensor = self.world.spawn_actor(seg_cam, left_spawn_point, attach_to=self.vehicle)
        self.actor_list.append(left_seg_sensor)
        left_seg_sensor.listen(lambda data: self.process_segmentation_img(data, 1))

        right_seg_sensor = self.world.spawn_actor(seg_cam, right_spawn_point, attach_to=self.vehicle)
        self.actor_list.append(right_seg_sensor)
        right_seg_sensor.listen(lambda data: self.process_segmentation_img(data, 2))

        col_sensor_bp = self.blueprint_library.find("sensor.other.collision")
        col_sensor = self.world.spawn_actor(col_sensor_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(col_sensor)
        col_sensor.listen(lambda event: self.collision_data(event))

        line_sensor_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        line_sensor = self.world.spawn_actor(line_sensor_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(line_sensor)
        line_sensor.listen(lambda event: self.on_invasion(event))

        while self.front_seg_camera is None or self.left_seg_camera is None or self.right_seg_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

        return self.get_state()

    def get_state(self):
        lane_left_dis = 0
        lane_left_type = [0, 0, 0, 1]
        lane_right_dis = 0
        lane_right_type = [0, 0, 0, 1]

        waypoint = self.client.get_world().get_map().get_waypoint(self.vehicle.get_location(), project_to_road=True)
        vehicle_location = self.vehicle.get_location()

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
        lane_delta_angle = deltaAngle

        left_lane_point = waypoint.get_left_lane()
        right_lane_point = waypoint.get_right_lane()
        if left_lane_point is not None and right_lane_point is not None:
            left_lane_location = left_lane_point.transform.location
            right_lane_location = right_lane_point.transform.location

            left_distance = math.sqrt(math.pow((vehicle_location.x - left_lane_location.x), 2)
                                      + math.pow((vehicle_location.y - left_lane_location.y), 2)
                                      + math.pow((vehicle_location.z - left_lane_location.z), 2))
            right_distance = math.sqrt(math.pow((vehicle_location.x - right_lane_location.x), 2)
                                       + math.pow((vehicle_location.y - right_lane_location.y), 2)
                                       + math.pow((vehicle_location.z - right_lane_location.z), 2))
            lane_left_dis = left_distance
            lane_right_dis = right_distance
            lane_left_type = self.lane_type_to_onehot_Vector(waypoint.left_lane_marking.type)
            lane_right_type = self.lane_type_to_onehot_Vector(waypoint.left_lane_marking.type)

        # normalization
        lane_width = waypoint.lane_width * 2

        lane_left_dis /= lane_width
        lane_right_dis /= lane_width

        lane_left_dis = min(1.0, lane_left_dis)
        lane_right_dis = min(1.0, lane_right_dis)

        lane_delta_angle /= 180

        # left_dis (1) right_dis (1) delta angle (1) left lane type (4) right lane type (4)
        return [self.left_seg_camera, self.right_seg_camera, self.front_seg_camera,
                [lane_left_dis, lane_right_dis, lane_delta_angle, lane_left_type, lane_right_type]]

    def collision_data(self, event):
        self.collision_hist.append(event)

    def on_invasion(self, event):
        self.invasion_hist.append(event)

    def process_segmentation_img(self, image, dir):
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data) / 255
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        if SHOW_CAM and dir == 0:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        if dir == 0:
            self.front_seg_camera = i3
        elif dir == 1:
            self.left_seg_camera = i3
        elif dir == 2:
            self.right_seg_camera = i3

    def lane_type_to_onehot_Vector(self, lane_type):
        if lane_type is carla.LaneMarkingType.Broken:
            return [1.0, 0.0, 0.0, 0.0]
        elif lane_type is carla.LaneMarkingType.SolidSolid:
            return [0.0, 1.0, 0.0, 0.0]
        elif lane_type is carla.LaneMarkingType.Solid:
            return [0.0, 0.0, 1.0, 0.0]
        else:
            return [0.0, 0.0, 0.0, 1.0]

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
            return self.get_state(), 0, False, None

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        reward = 0

        # 충돌 reward
        if len(self.collision_hist) != 0:
            done = True
            reward -= 500
        else:
            # waypoint = self.client.get_world().get_map().get_waypoint(self.vehicle.get_location(), project_to_road=True)
            #
            vehicle_location = self.vehicle.get_location()
            # waypoint_location = waypoint.transform.location
            # distance = math.sqrt(math.pow((vehicle_location.x - waypoint_location.x), 2)
            #                      + math.pow((vehicle_location.y - waypoint_location.y), 2)
            #                      + math.pow((vehicle_location.z - waypoint_location.z), 2))
            #
            # # 도로 중심으로 부터 거리 & 도로 방향과 차이 각도 reward
            # distance = int(distance * 10)
            #
            # reward -= distance

            # 차량 속도에 따른 reward
            velocity_value = int(kmh * kmh / 100 - 1)
            reward += velocity_value

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

        return self.get_state(), reward, done, None

    def destroy_actors(self):
        print(f'destroy actors({len(self.actor_list)})')
        # self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicle_list])

        for actor in self.actor_list:
            actor.destroy()

        self.actor_list.clear()
        self.vehicle_list.clear()
