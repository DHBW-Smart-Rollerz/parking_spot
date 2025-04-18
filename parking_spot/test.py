from camera_preprocessing.transformation.coordinate_transform import CoordinateTransform
import square_points

transform = CoordinateTransform()
list = [(14, 242), (140, 170), (206, 133), (245, 110)]
RT = []
for i in list:
    RT.append(transform.camera_to_world(i))
print(RT)
print(
    square_points.distance(RT[0], RT[1]),
    square_points.distance(RT[1], RT[2]),
    square_points.distance(RT[2], RT[3]),
)
ret = square_points.find_collinear_triplets(RT)
for i in ret:
    print(transform.world_to_camera(i))
