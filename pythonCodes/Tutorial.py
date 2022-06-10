import carla
import random

client = carla.Client('localhost', 2000)
world = client.get_world()
client.load_world('Town01')

# 카메라 위치 설정
spectator = world.get_spectator()

transform = spectator.get_transform()

location = transform.location
rotation = transform.rotation

spectator.set_transform(carla.Transform())

# 차량 생성
vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')

spawn_points = world.get_map().get_spawn_points()

for i in range(0, 50):
    world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

ego_vehicle = world.spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))


camera_init_trans = carla.Transform(carla.Location(z=1.5))

camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

camera.listen(lambda image: image.save_to_disk('out/%06d.png' % image.frame))