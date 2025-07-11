import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List


class PersonTracker:

    def __init__(self):
        self.people: Dict[int, dict] = {}
        self.next_id: int = 0
        self.max_distance: int = 50

    def update(self, detections: List[Tuple[int, int, int, int]]) -> Dict[int, dict]:
        current_ids = []

        for (x, y, w, h) in detections:
            center = self._calculate_center(x, y, w, h)
            matched_id = self._find_matching_person(center)

            if matched_id is not None:
                self._update_existing_person(matched_id, x, y, w, h, center)
                current_ids.append(matched_id)
            else:
                new_id = self._add_new_person(x, y, w, h, center)
                current_ids.append(new_id)

        self._remove_old_people(current_ids)
        return self.people

    def _calculate_center(self, x: int, y: int, w: int, h: int) -> Tuple[int, int]:
        return (x + w // 2, y + h // 2)

    def _find_matching_person(self, center: Tuple[int, int]) -> int:
        min_dist = float('inf')
        matched_id = None

        for person_id, person_data in self.people.items():
            last_center = person_data['last_center']
            dist = np.sqrt((center[0] - last_center[0]) ** 2 + (center[1] - last_center[1]) ** 2)

            if dist < min_dist and dist < self.max_distance:
                min_dist = dist
                matched_id = person_id

        return matched_id

    def _update_existing_person(self, person_id: int, x: int, y: int, w: int, h: int, center: Tuple[int, int]):
        self.people[person_id].update({
            'box': (x, y, w, h),
            'last_center': center,
            'last_seen': datetime.now()
        })

    def _add_new_person(self, x: int, y: int, w: int, h: int, center: Tuple[int, int]) -> int:
        self.people[self.next_id] = {
            'box': (x, y, w, h),
            'last_center': center,
            'first_seen': datetime.now(),
            'last_seen': datetime.now()
        }
        self.next_id += 1
        return self.next_id - 1

    def _remove_old_people(self, current_ids: List[int]):
        to_delete = [pid for pid in self.people if pid not in current_ids]
        for pid in to_delete:
            del self.people[pid]


class PeopleDetector:

    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.body_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        )
        self.face_tracker = PersonTracker()
        self.body_tracker = PersonTracker()

    def process_video_stream(self, video_source: str = 0, output_file: str = 'output.mp4'):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {video_source}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self._process_frame(frame)
                cv2.imshow('People Detection', frame)
                out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

    def _process_frame(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        bodies = self.body_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        tracked_faces = self.face_tracker.update(faces)
        tracked_bodies = self.body_tracker.update(bodies)

        self._draw_detections(frame, tracked_faces, (0, 255, 0), "Face")
        self._draw_detections(frame, tracked_bodies, (0, 0, 255), "Body")

        cv2.putText(frame, f"Faces: {len(tracked_faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Bodies: {len(tracked_bodies)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def _draw_detections(self, frame: np.ndarray, detections: Dict[int, dict],
                         color: Tuple[int, int, int], label_prefix: str):
        for person_id, person_data in detections.items():
            x, y, w, h = person_data['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label_prefix} ID: {person_id}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


if __name__ == "__main__":
    detector = PeopleDetector()

    detector.process_video_stream(video_source='video.mp4', output_file='output_video.mp4')