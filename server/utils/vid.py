import boto3
import cv2
import time
import os

boto3.setup_default_session(profile_name='rekognitioninarvo')
rekognition_client = boto3.client('rekognition')

# Public folder path (change as needed)
PUBLIC_FOLDER = '/Users/shashankkarna/Downloads/neww/server/static/vid_results'

def download_video_from_s3(bucket_name, video_name, local_video_path):
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket_name, video_name, local_video_path)
    except Exception as e:
        print(f"Error downloading video from S3: {e}")
        return False
    return True

def analyze_video(bucket_name, video_name, collection_id):
    response = rekognition_client.start_face_search(
        Video={'S3Object': {'Bucket': bucket_name, 'Name': video_name}},
        CollectionId=collection_id,
        FaceMatchThreshold=90
    )
    return response['JobId']

def get_face_search_results(job_id):
    finished = False
    while not finished:
        response = rekognition_client.get_face_search(JobId=job_id)
        if response['JobStatus'] in ['SUCCEEDED', 'FAILED']:
            finished = True
        else:
            time.sleep(5)
    return response

def draw_bounding_boxes(frame, bounding_boxes, face_ids):
    for box, face_id in zip(bounding_boxes, face_ids):
        left = int(box['Left'] * frame.shape[1])
        top = int(box['Top'] * frame.shape[0])
        width = int(box['Width'] * frame.shape[1])
        height = int(box['Height'] * frame.shape[0])

        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
        cv2.putText(frame, face_id, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

def process_video(bucket_name, video_name, collection_id):
    local_video_path = os.path.join(PUBLIC_FOLDER, video_name)
    if not download_video_from_s3(bucket_name, video_name, local_video_path):
        return None

    job_id = analyze_video(bucket_name, video_name, collection_id)
    response = get_face_search_results(job_id)

    cap = cv2.VideoCapture(local_video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

   
    faces_data = {}
    for person in response['Persons']:
        frame_number = int(person['Timestamp'] * frame_rate / 1000)
        if 'FaceMatches' in person and person['FaceMatches']:
            faces_data[frame_number] = {
                'BoundingBoxes': [face_match['Face']['BoundingBox'] for face_match in person['FaceMatches']],
                'FaceIds': [face_match['Face']['ExternalImageId'] for face_match in person['FaceMatches']]
            }
        else:
            faces_data[frame_number] = {
                'BoundingBoxes': [],
                'FaceIds': []
            }

    output_filename = f"processed_{video_name}"
    output_path = os.path.join(PUBLIC_FOLDER, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number in faces_data:
            bounding_boxes = faces_data[frame_number]['BoundingBoxes']
            face_ids = faces_data[frame_number]['FaceIds']
            draw_bounding_boxes(frame, bounding_boxes, face_ids)

        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()

    return output_filename


# Example usage:
# process_video('rekognitioninarvo', 'trimmed.MOV', 'rekognitioninarvo')
