import numpy as np
import torch
import torch.nn.functional as F
from torch_ import tapir_model
from utils import transforms as tfs
import glob
import os
import cv2
from collections import deque

from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter
from image_similarity.cal_similarity import cal_similar

from PIL import Image
from sentence_transformers import util
import open_clip
import torchvision.transforms as transforms

# device=torch.device('cuda')

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.cls = -1
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(cls, score)

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, score):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += score
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, score])
                self.cls = cls
        else:
            self.cls_hist.append([cls, score])
            self.cls = cls

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        self.update_cls(new_track.cls, new_track.score)

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        # update new bbox axis from yolo
        self._tlwh = new_tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.update_cls(new_track.cls, new_track.score)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        # if self.mean is None:
        #     return self._tlwh.copy()
        # ret = self.mean[:4].copy()
        # ret[:2] -= ret[2:] / 2
        # return ret

        # We disable the kalman filter output
        return self._tlwh.copy()

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

def preprocess_frames(frames):
  """Preprocess frames to model inputs.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

  Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
  """
  frames = frames.float()
  frames = frames / 255 * 2 - 1
  return frames



def postprocess_occlusions(occlusions, expected_dist):
  visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.1
  return visibles


def inference(frames, query_points, model):
  # Preprocess video to match model inputs format
  frames = preprocess_frames(frames)
  num_frames, height, width = frames.shape[0:3]
  query_points = query_points.float()
  frames, query_points = frames[None], query_points[None] #增加一維

  # Model inference
  
  outputs = model(frames, query_points)
  tracks, occlusions, expected_dist = outputs['tracks'][0], outputs['occlusion'][0], outputs['expected_dist'][0]
  # track : shape = (num_point,num_frams,2) , 2=[x,y]
  # occlusions expected_dist : shape = (num_point,num_frams)

  # Binarize occlusions
  visibles = postprocess_occlusions(occlusions, expected_dist) #看點是否還在畫面中吧
  return tracks, visibles

class CLIP_SIMILE(object):
    def __init__(self, args, device, frame_rate=30):
        self.device = device
        self.model_track = tapir_model.TAPIR(pyramid_level=1)
        self.model_track.load_state_dict(torch.load('checkpoints//bootstapir_checkpoint_v2.pt'))
        self.model_track = self.model_track.to(self.device)
        self.last_detection = []
        self.save_image = [[]]
        self.resize_height, self.resize_width=512, 512

        self.similar_model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
        self.similar_model.to(self.device)

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()
        self.img_connect = [[]]
        self.frame_id = 0
        self.threshold = 0.7
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    
    def update(self, output_results, img, frameID=0):
        self.frame_id += 1
        # print("frame_id",self.frame_id)
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        # print("frame_id:",self.frame_id)
        # print("output:", len(output_results))
        if len(output_results):
            bboxes = output_results[:, :4]
            scores = output_results[:, 4]
            classes = output_results[:, 5]
            features = output_results[:, 6:]
            
            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]
            features = output_results[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]
            features_keep = features[remain_inds]
        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        # # plot original bbox from yolo
        # img_debug = img.copy()
        # for tlbr in dets:
        #     plot_one_box(tlbr, img_debug, label='car', line_thickness=2)

        # cv2.imwrite(f'frame{frameID}.jpg', img_debug)
        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, f) for
                              (tlbr, s, c, f) in zip(dets, scores_keep, classes_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                              (tlbr, s, c) in zip(dets, scores_keep, classes_keep)]
        else:
            detections = []
        img = np.array(img)
        image_raw = img.copy()
        
        # print(dets)
        "Match"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.resize_height, self.resize_width))
        if len(self.img_connect)==2:
          self.img_connect[1] = img
          self.save_image[1] = image_raw
        else:
          if self.frame_id == 1:
            self.img_connect[0] = img
            self.save_image[0] = image_raw
          elif self.frame_id == 2:
            self.img_connect.append(img)
            self.save_image.append(image_raw)

        if self.frame_id == 1:
          # Initialize and activate all detections as new tracks
          for det in detections:
              det.activate(self.kalman_filter, self.frame_id)
              self.tracked_stracks.append(det)  # Add to tracked tracks
          self.img_connect[0] = img  # Save the first frame if needed
          self.last_detection = detections
          return detections  # Return the activated tracks

        elif self.frame_id != 1:
          object_similar_score = []
          for detection in detections:
              # print("detection:",detection)
              # Convert current detection bbox to the format [x1, y1, x2, y2]
              current_bbox = self.tlwh_to_xyxy(detection.tlwh)
            #   print("current_bbox:",current_bbox)
              
              matched_ind = None  # Flag to check if a detection has been matched to a track
              max_point = 0
              match_status = False
              similar_score = []
              for i, last_detection in enumerate(self.last_detection):
                  # print("last_detection:",last_detection)
                  xyxy = self.tlwh_to_xyxy(last_detection.tlwh)
                  match_xyxy = [self.tlwh_to_xyxy(last_detection.tlwh), self.tlwh_to_xyxy(detection.tlwh)]
                  r_match = self.extract_image_patches(self.save_image, match_xyxy)
                  sim_score = self.generateScore(r_match)
                  similar_score.append(sim_score)

              object_similar_score.append(similar_score)
          

          remember_dict = {}
          lost_ids = []
          for ind in range(len(self.last_detection)):
            # if ind not in sim_ind_list:
            #點追蹤
            max_index = max((index for index, score in enumerate(object_similar_score) if score[ind] >= self.threshold), key=lambda index: object_similar_score[index][ind], default=None)
            # max_index = max((index for index, score in enumerate(object_point_score)), key=lambda index: object_point_score[index][ind], default=None)
            
            if max_index != None and detections[max_index].track_id != 0:
                last_information = remember_dict.get(str(max_index), "Key not found")
                # print("last_conf:",last_conf)
                if last_information != "Key not found":
                    last_conf = last_information[0]
                    if object_similar_score[max_index][ind] > last_conf:
                        last_ind = last_information[1]
                        lost_ids.append(last_ind)
                        detections[max_index].track_id = self.last_detection[ind].track_id
                    else:
                        lost_ids.append(ind)

            if max_index != None and detections[max_index].track_id == 0:
                remember_dict[str(max_index)] = [object_similar_score[max_index][ind], ind]
                detections[max_index].track_id = self.last_detection[ind].track_id

          
          for detection in detections:
            if detection.track_id == 0:
                detection.track_id = detection.next_id()

          # Update last_detection for the next frame
          self.last_detection = detections
          self.img_connect[0] = img
          self.save_image[0] = image_raw
          return detections

        # elif self.frame_id != 1:
        #   for detection in detections:
        #     xyxy = self.tlwh_to_xyxy(detection.tlwh)
        #     new_points = self.generate_points(xyxy, 4, 4)
        #     print("new_points:", new_points)
        #     query_points = self.convert_select_points_to_query_points(0, new_points)
            
        #     image_concat = torch.tensor(np.array(self.img_connect)).to(device)
        #     query_points = torch.tensor(query_points).to(device)
        #     tracks, visibles = inference(image_concat, query_points, self.model_track)
            
        #     track_points = tracks[:, :, 1].view(-1, 2).cpu().numpy()
        #     print("tracks:",tracks)
        #     # print(tracks.shape)
        #     print("track_points:",track_points)

        #     self.img_connect[0] = img
        #   return detections

    def extract_image_patches(self, two_images, bboxes):
        bboxes = np.round(bboxes).astype(np.int32)
        
        # Check if the length of images and bboxes are both 2
        assert len(two_images) == 2, "There should be exactly two images."
        assert len(bboxes) == 2, "There should be exactly two bounding boxes."
        
        patches = []
        for i in range(2):
            image = two_images[i]
            box = bboxes[i]
            patch = image[box[1]:box[3], box[0]:box[2], :]
            # if i == 0:
            #     cv2.imwrite("image1.jpg",patch)
            # else:
            #     cv2.imwrite("image2.jpg",patch)
            patches.append(patch)
        
        return patches
    # def extract_image_patches(self, image, bboxes):
    #         bboxes = np.round(bboxes).astype(np.int32)
    #         patches = [image[box[1]:box[3], box[0]:box[2],:] for box in bboxes]    
    #         #bboxes = clip_boxes(bboxes, image.shape)
    #         return patches
        
    def convert_select_points_to_query_points(self, frame, points):
      """Convert select points to query points.

      Args:
        points: [num_points, 2], in [x, y]
      Returns:
        query_points: [num_points, 3], in [t, y, x]
      """
      points = np.stack(points)
      query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
      query_points[:, 0] = frame
      query_points[:, 1] = points[:, 1]
      query_points[:, 2] = points[:, 0]
      return query_points

    def generate_points(self, rect_coords, rows, cols):
      x1, y1, x2, y2 = rect_coords
      points = []

      # 計算水平和垂直方向的間隔
      x_interval = (x2 - x1) / (cols - 1) if cols > 1 else 0
      y_interval = (y2 - y1) / (rows - 1) if rows > 1 else 0

      # 生成點
      for i in range(rows):
          for j in range(cols):
              x = x1 + j * x_interval
              y = y1 + i * y_interval
              points.append((x, y))
      return points

    def tlwh_to_xyxy(self, tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[2] = ret[0] + ret[2]
        ret[3] = ret[1] + ret[3]
        return ret

    def check_points_in_bbox(self, points, bbox):
      """
      Check if at least 8 out of the given points are within the specified bounding box.

      Args:
          points (np.array): Array of points with shape (num_points, 2) where each row is (x, y).
          bbox (list): The bounding box [x1, y1, x2, y2].

      Returns:
          bool: True if at least 8 points are inside the bbox, False otherwise.
      """
      x1, y1, x2, y2 = bbox
      # print("bbox:",bbox)
      # print("points:",points)
      count = 0
      for x, y in points:
          if x1 <= x <= x2 and y1 <= y <= y2:
              count += 1
      return count
    #       if count >= 3:
    #           # print("count:",count)
    #           return True, count
    #   # print("count:",count)
    #   return False, count

    def convert_coordinates(self, data, old_width, old_height, new_width, new_height):
        # 確保輸入是 torch.Tensor
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        # 設置裝置
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data = data.to(self.device)
        
        # 計算寬度和高度的縮放比例
        width_ratio = new_width / old_width
        height_ratio = new_height / old_height
        
        # 計算新的座標
        scaled_data = data * torch.tensor([width_ratio, height_ratio], device=self.device)
        
        return scaled_data

    def compute_iou(self, box1, box2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters:
        - box1, box2: Numpy arrays of shape [4] -> [x1, y1, x2, y2]

        Returns:
        - iou: Float, the IoU ratio
        """
        # Determine the coordinates of the intersection rectangle
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        # Calculate the area of intersection rectangle
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate the area of both bounding boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Compute the Union area by using the formula: Union(A,B) = A + B - Intersect(A,B)
        union_area = box1_area + box2_area - intersection_area
        # Compute the IoU by dividing the intersection area by the union area
        iou = intersection_area / union_area
        return iou
    
    def find_indices_above_threshold(self, object_point_score, threshold):
        exceed_indices = []
        for i, row in enumerate(object_point_score):
            count_above_threshold = sum(1 for value in row if value > threshold)
            if count_above_threshold >= 2:
                for j, value in enumerate(row):
                    if value > threshold:
                        exceed_indices.append((i, j))
        return exceed_indices

    def imageEncoder(self, img):
        img1 = Image.fromarray(img).convert('RGB')
        img1 = self.preprocess(img1).unsqueeze(0).to(self.device)
        img1 = self.similar_model.encode_image(img1)
        return img1

    def generateScore(self, r_match):
        test_img = r_match[0]
        data_img = r_match[1]
        img1 = self.imageEncoder(test_img)
        img2 = self.imageEncoder(data_img)
        cos_scores = util.pytorch_cos_sim(img1, img2)
        score = round(float(cos_scores[0][0]), 2)
        return score
      
