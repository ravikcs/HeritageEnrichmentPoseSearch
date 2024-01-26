import mmpose

from mmpose.apis import MMPoseInferencer

#img_path = 'C:/Users/Administrator/Desktop/pose estimation/art/*.jpg'   
#replace this with your own image path

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer('vitpose-l')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
'''result_generator = inferencer(img_path, show=False, vis_out_dir='vis_results')
result = next(result_generator)'''

folder_path = 'C:/Users/Administrator/Desktop/pose estimation/dummydata/'

result_generator = inferencer(folder_path, show=False, vis_out_dir='vis_results', pred_out_dir = 'pred_results', draw_bbox = True, draw_heatmap = False)
results = [result for result in result_generator]
kp_dict = {}
for i in range(len(results)):
    pred_dict = results[i]['predictions'][0][0]
    print(f"***printing keypoints for {i} image***")
    print(len(pred_dict['keypoints']))
    print(pred_dict['keypoints'])
    kp_id = f"{i}"
    #print(f"***printing keypoints score for {i} image***")
    #print(len(pred_dict['keypoint_scores']))
    #print(pred_dict['keypoint_scores'])
    kp_dict[kp_id] = pred_dict['keypoints']
print("***printing kp_list***")
print(kp_dict)

'''import numpy as np
# Convert the list of lists to a NumPy array
array_of_lists = np.array(kp_list)
print(len(array_of_lists))
print(array_of_lists)
# Calculate pairwise Euclidean distances between the points
distances = {}
for i in range(len(array_of_lists)):
    for j in range(i+1, len(array_of_lists)):
        distance_between = f"{i}_{j}"
        dist = np.linalg.norm(array_of_lists[i] - array_of_lists[j])
        distances[distance_between] = dist

print("Pairwise Euclidean distances:")
print(distances)'''

'''import numpy as np
# Convert dictionary values (lists) to NumPy arrays for computation
poses_np = {k: np.array(v) for k, v in kp_dict.items()}

print(poses_np)

# Calculate Euclidean distance between corresponding keypoints for each pose pair
distances = {}
for pose1_keypoints in poses_np:
    print("printing pose1_keypoints")
    print(pose1_keypoints)
    distances[pose1_keypoints] = {}
    for pose2_keypoints in poses_np:
        print("printing pose2_keypoints")
        print(pose2_keypoints)
        distance = np.linalg.norm(poses_np[pose1_keypoints] - poses_np[pose2_keypoints])
        distances[pose1_keypoints][pose2_keypoints] = distance

print("priniting distance dict")
print(distances)
# Display the distances between poses
print("Euclidean Distances between Poses:")
for pose1_key, pose1_distances in distances.items():
    for pose2_key, distance in pose1_distances.items():
        print(f"Distance between {pose1_key} and {pose2_key}: {distance}")'''