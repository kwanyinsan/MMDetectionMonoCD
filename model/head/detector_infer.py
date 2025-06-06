import torch
import pdb
import math
import random

from model.layers.iou_loss import get_corners
from torch import nn
from shapely.geometry import Polygon
from torch.nn import functional as F


from model.anno_encoder import Anno_Encoder
from model.layers.utils import (
	nms_hm,
	select_topk,
	select_point_of_interest,
)

from model.layers.utils import Converter_key2channel
from engine.visualize_infer import box_iou, box_iou_3d, box3d_to_corners

def make_post_processor(cfg):
	anno_encoder = Anno_Encoder(cfg)
	key2channel = Converter_key2channel(keys=cfg.MODEL.HEAD.REGRESSION_HEADS, channels=cfg.MODEL.HEAD.REGRESSION_CHANNELS)
	postprocessor = PostProcessor(cfg=cfg, anno_encoder=anno_encoder, key2channel=key2channel)
	
	return postprocessor

class PostProcessor(nn.Module):
	def __init__(self, cfg, anno_encoder, key2channel):
		
		super(PostProcessor, self).__init__()

		self.anno_encoder = anno_encoder
		self.key2channel = key2channel

		self.det_threshold = cfg.TEST.DETECTIONS_THRESHOLD
		self.max_detection = cfg.TEST.DETECTIONS_PER_IMG		
		self.eval_dis_iou = cfg.TEST.EVAL_DIS_IOUS
		self.eval_depth = cfg.TEST.EVAL_DEPTH

		self.output_width = cfg.INPUT.WIDTH_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
		self.output_height = cfg.INPUT.HEIGHT_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
		self.output_depth = cfg.MODEL.HEAD.OUTPUT_DEPTH

		self.pred_2d = cfg.TEST.PRED_2D
		self.vis = cfg.TEST.VIS
		self.vis_all = cfg.TEST.VIS_ALL

		self.compensatory_test = cfg.TEST.COMPENSATORY_TEST
		self.flip_type = cfg.TEST.FLIP_TYPE

		self.pred_direct_depth = 'depth' in self.key2channel.keys
		self.depth_with_uncertainty = 'depth_uncertainty' in self.key2channel.keys
		self.regress_keypoints = 'corner_offset' in self.key2channel.keys
		self.keypoint_depth_with_uncertainty = 'corner_uncertainty' in self.key2channel.keys
		self.compute_compensated_depth = 'compensated_depth_uncertainty' in self.key2channel.keys
		self.pred_y3d = 'y3d_offset' in self.key2channel.keys
		self.use_ideal_y3d = cfg.FIXED_Y3D

		self.use_ground_plane = cfg.USE_GROUND_PLANE
		self.pred_ground_plane = cfg.MODEL.HEAD.PRED_GROUND_PLANE
		self.pred_multi_y3d = cfg.MODEL.HEAD.PRED_MULTI_Y3D

		# use uncertainty to guide the confidence
		self.uncertainty_as_conf = cfg.TEST.UNCERTAINTY_AS_CONFIDENCE

	def prepare_targets(self, targets, test):
		pad_size = torch.stack([t.get_field("pad_size") for t in targets])
		calibs = [t.get_field("calib") for t in targets]
		size = torch.stack([torch.tensor(t.size) for t in targets])

		if test:
			target_varibales = dict(calib=calibs, size=size, pad_size=pad_size)
			if self.pred_ground_plane:
				target_varibales['horizon_state'] = [t.get_field("horizon_state") for t in targets]
			return target_varibales

		cls_ids = torch.stack([t.get_field("cls_ids") for t in targets])
		# regression locations (in pixels)
		target_centers = torch.stack([t.get_field("target_centers") for t in targets])
		# 3D infos
		dimensions = torch.stack([t.get_field("dimensions") for t in targets])
		rotys = torch.stack([t.get_field("rotys") for t in targets])
		locations = torch.stack([t.get_field("locations") for t in targets])
		# offset_2D = torch.stack([t.get_field("offset_2D") for t in targets])
		offset_3D = torch.stack([t.get_field("offset_3D") for t in targets])
		# reg mask
		reg_mask = torch.stack([t.get_field("reg_mask") for t in targets])

		EL = torch.stack([t.get_field("EL") for t in targets])

		target_varibales = dict(pad_size=pad_size, calib=calibs, size=size, cls_ids=cls_ids, target_centers=target_centers, EL=EL,
							dimensions=dimensions, rotys=rotys, locations=locations, offset_3D=offset_3D, reg_mask=reg_mask,)

		if self.use_ground_plane:
			target_varibales['ground_plane'] = torch.stack([t.get_field("ground_plane") for t in targets])

		if self.pred_ground_plane:
			target_varibales['horizon_state'] = [t.get_field("horizon_state") for t in targets]
			target_varibales['ground_plane'] = torch.stack([t.get_field("ground_plane") for t in targets])

		return target_varibales

	def forward(self, predictions, targets, features=None, test=False, refine_module=None):
		pred_heatmap, pred_regression = predictions['cls'], predictions['reg']
		batch = pred_heatmap.shape[0]

		target_varibales = self.prepare_targets(targets, test=test)
		calib, pad_size = target_varibales['calib'], target_varibales['pad_size']
		img_size = target_varibales['size']

		# evaluate the disentangling IoU for each components in (location, dimension, orientation)
		dis_ious = self.evaluate_3D_detection(target_varibales, pred_regression) if self.eval_dis_iou else None

		# evaluate the accuracy of predicted depths
		depth_errors = self.evaluate_3D_depths(target_varibales, pred_regression) if self.eval_depth else None

		# max-pooling as nms for heat-map
		heatmap = nms_hm(pred_heatmap)
		visualize_preds = {'heat_map': pred_heatmap.clone()}

		# ─── Added for MMDetection usage ────────────────────────────────────
		# 1) index of the 'depth' regression head
		depth_ch     = self.key2channel('depth')
		# 2) raw per-pixel offsets (B×H×W)
		raw_offset   = pred_regression[:, depth_ch, :, :]
		# 3) decode to metric depth (meters, B×H×W)
		dense_depth  = self.anno_encoder.decode_depth(raw_offset)
		# 4) add to visualize_preds so downstream code can save/display it
		visualize_preds['depth_map'] = dense_depth
		# ────────────────────────────────────────────────────────────────────

		if self.pred_ground_plane:
			visualize_preds['horizon_heat_map'] = predictions['horizon'].clone()

		# select top-k of the predicted heatmap
		scores, indexs, clses, ys, xs = select_topk(heatmap, K=self.max_detection)

		pred_bbox_points = torch.cat([xs.view(-1, 1), ys.view(-1, 1)], dim=1)
		pred_regression_pois = select_point_of_interest(batch, indexs, pred_regression).view(-1, pred_regression.shape[1])

		# thresholding with score
		scores = scores.view(-1)
		valid_mask = scores >= self.det_threshold

		# no valid predictions
		if valid_mask.sum() == 0:
			result = scores.new_zeros(0, 14)
			visualize_preds['keypoints'] = scores.new_zeros(0, 20)
			visualize_preds['proj_center'] = scores.new_zeros(0, 2)
			eval_utils = {'dis_ious': dis_ious, 'depth_errors': depth_errors, 'vis_scores': scores.new_zeros(0),
					'uncertainty_conf': scores.new_zeros(0), 'estimated_depth_error': scores.new_zeros(0)}
			
			return result, eval_utils, visualize_preds

		scores = scores[valid_mask]
		
		clses = clses.view(-1)[valid_mask]
		pred_bbox_points = pred_bbox_points[valid_mask]
		pred_regression_pois = pred_regression_pois[valid_mask]
		pred_y3d = None

		pred_2d_reg = F.relu(pred_regression_pois[:, self.key2channel('2d_dim')])
		pred_offset_3D = pred_regression_pois[:, self.key2channel('3d_offset')]

		pred_orientation = torch.cat((pred_regression_pois[:, self.key2channel('ori_cls')], pred_regression_pois[:, self.key2channel('ori_offset')]), dim=1)
		visualize_preds['proj_center'] = pred_bbox_points + pred_offset_3D

		pred_box2d = self.anno_encoder.decode_box2d_fcos(pred_bbox_points, pred_2d_reg, pad_size, img_size)

		pred_dimensions_offsets = pred_regression_pois[:, self.key2channel('3d_dim')]
		pred_dimensions = self.anno_encoder.decode_dimension(clses, pred_dimensions_offsets)

		if self.pred_direct_depth:
			pred_depths_offset = pred_regression_pois[:, self.key2channel('depth')].squeeze(-1)
			pred_direct_depths = self.anno_encoder.decode_depth(pred_depths_offset)

			if self.compensatory_test and self.flip_type == 'direct':
				if random.random() >= 0.5:
					target_center = target_varibales['target_centers'].float()[target_varibales['reg_mask']]
					target_depth = target_varibales['locations'][target_varibales['reg_mask']][:, 2]
					if len(target_center) > 0:
						dist = torch.sqrt(torch.sum((pred_bbox_points[:, None] - target_center) ** 2, axis=-1))
						nearest_index = torch.argmin(dist, axis=-1)
						target_depth = target_depth[nearest_index]
						pred_direct_depths = target_depth - (pred_direct_depths - target_depth)
						# add a random perturbation ranging from -A to A to pred_direct_depths
						# A = 2.0
						# pred_direct_depths += (random.random() - 0.5) * 2 * A

		if self.depth_with_uncertainty:
			pred_direct_uncertainty = pred_regression_pois[:, self.key2channel('depth_uncertainty')].exp()
			visualize_preds['depth_uncertainty'] = pred_regression[:, self.key2channel('depth_uncertainty'), ...].squeeze(1)

		if self.regress_keypoints:
			pred_keypoint_offset = pred_regression_pois[:, self.key2channel('corner_offset')]
			pred_keypoint_offset = pred_keypoint_offset.view(-1, 10, 2)

			if self.keypoint_depth_with_uncertainty:
				pred_keypoint_uncertainty = pred_regression_pois[:, self.key2channel('corner_uncertainty')].exp()
				# solve depth from estimated key-points
				pred_keypoints_depths = self.anno_encoder.decode_depth_from_keypoints_batch(pred_keypoint_offset, pred_dimensions, calib)
				visualize_preds['keypoints'] = pred_keypoint_offset
				if self.compensatory_test and 'key' in self.flip_type:
					if random.random() >= 0:
						target_center = target_varibales['target_centers'].float()[target_varibales['reg_mask']]
						target_depth = target_varibales['locations'][target_varibales['reg_mask']][:, 2]
						if len(target_center) > 0:
							dist = torch.sqrt(torch.sum((pred_bbox_points[:, None] - target_center) ** 2, axis=-1))
							nearest_index = torch.argmin(dist, axis=-1)
							target_depth = target_depth[nearest_index]
							if self.flip_type == 'key0':
								pred_keypoints_depths[:, 0] = target_depth - (pred_keypoints_depths[:, 0] - target_depth)
							elif self.flip_type == 'key1':
								pred_keypoints_depths[:, 1] = target_depth - (pred_keypoints_depths[:, 1] - target_depth)
							elif self.flip_type == 'key2':
								pred_keypoints_depths[:, 2] = target_depth - (pred_keypoints_depths[:, 2] - target_depth)
		if self.compensatory_test and self.flip_type == 'multiple':
			if random.random() >= 0.5:
				target_center = target_varibales['target_centers'].float()[target_varibales['reg_mask']]
				target_depth = target_varibales['locations'][target_varibales['reg_mask']][:, 2]
				if len(target_center) > 0:
					dist = torch.sqrt(torch.sum((pred_bbox_points[:, None] - target_center) ** 2, axis=-1))
					nearest_index = torch.argmin(dist, axis=-1)
					target_depth = target_depth[nearest_index]
					# 1
					pred_direct_depths = target_depth - (pred_direct_depths - target_depth)
					# 2
					pred_keypoints_depths[:, 0] = target_depth - (pred_keypoints_depths[:, 0] - target_depth)
					# 3
					pred_keypoints_depths[:, 1] = target_depth - (pred_keypoints_depths[:, 1] - target_depth)
					# 4
					pred_keypoints_depths[:, 2] = target_depth - (pred_keypoints_depths[:, 2] - target_depth)

		###### get y3d ######
		if self.pred_y3d:
			pred_y3d_offset = pred_regression_pois[:, self.key2channel('y3d_offset')]
			pred_y3d = self.anno_encoder.decode_y3d(clses, pred_y3d_offset)

		if self.use_ground_plane:
			pred_y3d = self.anno_encoder.decode_y3d_from_ground_plane(target_varibales['ground_plane'], pred_bbox_points,
																	pred_keypoint_offset, calib, pad_size)

		if self.pred_ground_plane:
			pred_horizon_heatmaps = predictions['horizon']
			pred_ground_plane = self.anno_encoder.decode_ground_plane_from_heatmap(pred_horizon_heatmaps,
																				   target_varibales['horizon_state'],
																				   calib,
																				   pad_size,
																				   )
			decode_y3d_func = self.anno_encoder.decode_multi_y3d_from_ground_plane if self.pred_multi_y3d else self.anno_encoder.decode_y3d_from_ground_plane
			pred_y3d = decode_y3d_func(pred_ground_plane,
										pred_bbox_points,
										pred_keypoint_offset, calib, pad_size)

		if self.compute_compensated_depth:
			if pred_y3d is not None:
				decode_compensated_depths_func = self.anno_encoder.decode_depth_from_roof_and_bottom_multi_y3d if self.pred_multi_y3d else self.anno_encoder.decode_depth_from_roof_and_bottom
				pred_compensated_depths = decode_compensated_depths_func(pred_y3d, pred_bbox_points, pred_keypoint_offset, pred_dimensions, calib, pad_size, None)
			else:
				if self.use_ideal_y3d:
					ideal_y3d = torch.ones(pred_bbox_points.shape[0], device='cuda:0') * 1.65
					EL = ideal_y3d
				else:
					target_center = target_varibales['target_centers'].float()
					reg_mask = target_varibales['reg_mask'].view(-1).bool()
					target_center = target_center[0][reg_mask]
					target_EL = target_varibales['EL'].view(-1)[reg_mask]
					if len(target_center) == 0:
						EL = torch.ones(pred_bbox_points.shape[0], device='cuda:0') * 1.65
					else:
						# pred_bbox_points[N, 2], target_center[M, 2]
						# match the closest target_center for N pred_bbox_points
						dist = torch.sqrt(torch.sum((pred_bbox_points[:, None] - target_center) ** 2, axis=-1))
						nearest_index = torch.argmin(dist, axis=-1)
						EL = target_EL[nearest_index]
				pred_compensated_depths = self.anno_encoder.decode_depth_from_roof_and_bottom(EL, pred_bbox_points, pred_keypoint_offset, pred_dimensions, calib, pad_size, None)

			pred_compensated_depths_uncertainty = pred_regression_pois[:, self.key2channel('compensated_depth_uncertainty')].exp()

		estimated_depth_error = None

		if self.output_depth == 'direct':
			pred_depths = pred_direct_depths

			if self.depth_with_uncertainty: estimated_depth_error = pred_direct_uncertainty.squeeze(dim=1)

		elif self.output_depth == 'direct_and_key0':
			pred_combined_depths = torch.stack([pred_direct_depths, pred_keypoints_depths[:, 0]], dim=1)
			pred_combined_uncertainty = torch.stack([pred_direct_uncertainty.squeeze(dim=1), pred_keypoint_uncertainty[:, 0]], dim=1)
			depth_weights = 1 / pred_combined_uncertainty
			depth_weights = depth_weights / depth_weights.sum(dim=1, keepdim=True)
			pred_depths = (pred_combined_depths * depth_weights).sum(dim=1)
			estimated_depth_error = torch.sum(pred_combined_uncertainty * depth_weights, dim=1)

		elif self.output_depth == 'direct_key0_and_key1':
			pred_combined_depths = torch.stack([pred_direct_depths, pred_keypoints_depths[:, 0], pred_keypoints_depths[:, 1]], dim=1)
			pred_combined_uncertainty = torch.stack([pred_direct_uncertainty.squeeze(dim=1), pred_keypoint_uncertainty[:, 0], pred_keypoint_uncertainty[:, 1]], dim=1)
			depth_weights = 1 / pred_combined_uncertainty
			depth_weights = depth_weights / depth_weights.sum(dim=1, keepdim=True)
			pred_depths = (pred_combined_depths * depth_weights).sum(dim=1)
			estimated_depth_error = torch.sum(pred_combined_uncertainty * depth_weights, dim=1)
		
		elif self.output_depth.find('keypoints') >= 0:
			if self.output_depth == 'keypoints_avg':
				pred_depths = pred_keypoints_depths.mean(dim=1)
				if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty.mean(dim=1)

			elif self.output_depth == 'keypoints_center':
				pred_depths = pred_keypoints_depths[:, 0]
				if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty[:, 0]

			elif self.output_depth == 'keypoints_02':
				pred_depths = pred_keypoints_depths[:, 1]
				if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty[:, 1]

			elif self.output_depth == 'keypoints_13':
				pred_depths = pred_keypoints_depths[:, 2]
				if self.keypoint_depth_with_uncertainty: estimated_depth_error = pred_keypoint_uncertainty[:, 2]

			else:
				raise ValueError

		elif self.output_depth.find('compensated') >= 0:
			if self.output_depth == 'compensated_center':
				pred_depths = pred_compensated_depths[:, 0]
				estimated_depth_error = pred_compensated_depths_uncertainty[:, 0]

			elif self.output_depth == 'compensated_02':
				pred_depths = pred_compensated_depths[:, 1]
				estimated_depth_error = pred_compensated_depths_uncertainty[:, 1]

			elif self.output_depth == 'compensated_13':
				pred_depths = pred_compensated_depths[:, 2]
				estimated_depth_error = pred_compensated_depths_uncertainty[:, 2]

		elif self.output_depth == 'raw_soft':
			pred_combined_depths = torch.cat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths), dim=1)
			pred_combined_uncertainty = torch.cat((pred_direct_uncertainty, pred_keypoint_uncertainty), dim=1)
			depth_weights = 1 / pred_combined_uncertainty
			depth_weights = depth_weights / depth_weights.sum(dim=1, keepdim=True)
			pred_depths = torch.sum(pred_combined_depths * depth_weights, dim=1)
			estimated_depth_error = torch.sum(depth_weights * pred_combined_uncertainty, dim=1)

		elif self.output_depth == 'soft' or self.output_depth == 'soft_square':
			if not self.pred_direct_depth and self.compute_compensated_depth:
				pred_combined_depths = torch.cat((pred_keypoints_depths, pred_compensated_depths), dim=1)
				pred_combined_uncertainty = torch.cat((pred_keypoint_uncertainty,
													   pred_compensated_depths_uncertainty), dim=1)

			elif self.compute_compensated_depth:
				pred_combined_depths = torch.cat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths,
												  pred_compensated_depths), dim=1)
				pred_combined_uncertainty = torch.cat((pred_direct_uncertainty, pred_keypoint_uncertainty,
													   pred_compensated_depths_uncertainty), dim=1)
			else:
				if self.pred_direct_depth and self.depth_with_uncertainty:
					pred_combined_depths = torch.cat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths), dim=1)
					pred_combined_uncertainty = torch.cat((pred_direct_uncertainty, pred_keypoint_uncertainty), dim=1)
				else:
					pred_combined_depths = pred_keypoints_depths.clone()
					pred_combined_uncertainty = pred_keypoint_uncertainty.clone()

			if self.output_depth == 'soft':
				depth_weights = 1 / pred_combined_uncertainty
			if self.output_depth == 'soft_square':
				depth_weights = 1 / pred_combined_uncertainty**2
			visualize_preds['min_uncertainty'] = depth_weights.argmax(dim=1)
			depth_weights = depth_weights / depth_weights.sum(dim=1, keepdim=True)
			pred_depths = torch.sum(pred_combined_depths * depth_weights, dim=1)

			# the uncertainty after combination
			if self.output_depth == 'soft':
				estimated_depth_error = torch.sum(depth_weights * pred_combined_uncertainty, dim=1)
			if self.output_depth == 'soft_square':
				estimated_depth_error = torch.sum(depth_weights**2 * pred_combined_uncertainty**2, dim=1)

		batch_idxs = pred_depths.new_zeros(pred_depths.shape[0]).long()
		pred_locations = self.anno_encoder.decode_location_flatten(pred_bbox_points, pred_offset_3D, pred_depths, calib, pad_size, batch_idxs)
		pred_rotys, pred_alphas = self.anno_encoder.decode_axes_orientation(pred_orientation, pred_locations)

		pred_locations[:, 1] += pred_dimensions[:, 1] / 2
		clses = clses.view(-1, 1)
		pred_alphas = pred_alphas.view(-1, 1)
		pred_rotys = pred_rotys.view(-1, 1)
		scores = scores.view(-1, 1)
		# change dimension back to h,w,l
		pred_dimensions = pred_dimensions.roll(shifts=-1, dims=1)

		# the uncertainty of depth estimation can reflect the confidence for 3D object detection
		vis_scores = scores.clone()
		if self.uncertainty_as_conf and estimated_depth_error is not None:
			uncertainty_conf = 1 - torch.clamp(estimated_depth_error, min=0.01, max=1)	
			scores = scores * uncertainty_conf.view(-1, 1)
		else:
			uncertainty_conf, estimated_depth_error = None, None

		# kitti output format
		result = torch.cat([clses, pred_alphas, pred_box2d, pred_dimensions, pred_locations, pred_rotys, scores], dim=1)

		# # filter the result out of 60m
		# mask = result[:, 11]<60
		# result = result[mask]
		# uncertainty_conf = uncertainty_conf[mask]
		# estimated_depth_error = estimated_depth_error[mask]
		# vis_scores = vis_scores[mask]

		if self.vis:
			pass
			# for i in range(result.shape[0]):
			# 	print('{}:soft_depth:{:.2f},soft_uncer:{:.2f},dir_depth:{:.2f},dir_uncer:{:.2f},center_depth:{:.2f},center_uncer:{:.2f},ground_depth:{:.2f},ground_uncer:{:.2f},'
			# 		  'height:{:.2f},scores:{:.2f},vis_scores:{:.2f}'.format(i+1, pred_depths[i], estimated_depth_error[i],
			# 													 pred_combined_depths[i,0], pred_combined_uncertainty[i,0],
			# 													 pred_combined_depths[i,1], pred_combined_uncertainty[i,1],
			# 													 pred_combined_depths[i,4], pred_combined_uncertainty[i,4],
			# 													 pred_box2d[i,3]-pred_box2d[i,1],scores[i,0],vis_scores[i,0]))
		if self.vis_all and self.output_depth == 'soft':
			print("prediction:")
			for i in range(result.shape[0]):
				print('{}:'.format(i+1))
				print('    soft_depth:{:.2f},uncer:{:.2f}'.format(pred_depths[i], estimated_depth_error[i]))

				# dirct, key, comp
				print('     dir_depth:{:.2f},uncer:{:.2f}'.format(pred_combined_depths[i,0], pred_combined_uncertainty[i,0]))
				print(' key_cen_depth:{:.2f},uncer:{:.2f}'.format(pred_combined_depths[i,1], pred_combined_uncertainty[i,1]))
				print('  key_02_depth:{:.2f},uncer:{:.2f}'.format(pred_combined_depths[i,2], pred_combined_uncertainty[i,2]))
				print('  key_13_depth:{:.2f},uncer:{:.2f}'.format(pred_combined_depths[i,3], pred_combined_uncertainty[i,3]))
				print('comp_cen_depth:{:.2f},uncer:{:.2f}'.format(pred_combined_depths[i,4], pred_combined_uncertainty[i,4]))
				print(' comp_02_depth:{:.2f},uncer:{:.2f}'.format(pred_combined_depths[i,5], pred_combined_uncertainty[i,5]))
				print(' comp_13_depth:{:.2f},uncer:{:.2f}'.format(pred_combined_depths[i,6], pred_combined_uncertainty[i,6]))

				# key, comp
				# print(' key_cen_depth:{:.2f},uncer:{:.2f}'.format(pred_combined_depths[i, 0],
				# 												  pred_combined_uncertainty[i, 0]))
				# print('  key_02_depth:{:.2f},uncer:{:.2f}'.format(pred_combined_depths[i, 1],
				# 												  pred_combined_uncertainty[i, 1]))
				# print('  key_13_depth:{:.2f},uncer:{:.2f}'.format(pred_combined_depths[i, 2],
				# 												  pred_combined_uncertainty[i, 2]))
				# print('comp_cen_depth:{:.2f},uncer:{:.2f}'.format(pred_combined_depths[i, 3],
				# 												  pred_combined_uncertainty[i, 3]))
				# print(' comp_02_depth:{:.2f},uncer:{:.2f}'.format(pred_combined_depths[i, 4],
				# 												  pred_combined_uncertainty[i, 4]))
				# print(' comp_13_depth:{:.2f},uncer:{:.2f}'.format(pred_combined_depths[i, 5],
				# 												  pred_combined_uncertainty[i, 5]))

		eval_utils = {'dis_ious': dis_ious, 'depth_errors': depth_errors, 'uncertainty_conf': uncertainty_conf,
					'estimated_depth_error': estimated_depth_error, 'vis_scores': vis_scores}
		
		return result, eval_utils, visualize_preds

	def get_oracle_depths(self, pred_bboxes, pred_clses, pred_combined_depths, pred_combined_uncertainty, target):
		calib = target.get_field('calib')
		pad_size = target.get_field('pad_size')
		pad_w, pad_h = pad_size

		valid_mask = target.get_field('reg_mask').bool()
		num_gt = valid_mask.sum()
		gt_clses = target.get_field('cls_ids')[valid_mask]
		gt_boxes = target.get_field('gt_bboxes')[valid_mask]
		gt_locs = target.get_field('locations')[valid_mask]

		gt_depths = gt_locs[:, -1]
		gt_boxes_center = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2

		iou_thresh = 0.5

		# initialize with the average values
		oracle_depth = pred_combined_depths.mean(dim=1)
		estimated_depth_error = pred_combined_uncertainty.mean(dim=1)

		for i in range(pred_bboxes.shape[0]):
			# find the corresponding object bounding boxes
			box2d = pred_bboxes[i]
			box2d_center = (box2d[:2] + box2d[2:]) / 2
			img_dis = torch.sum((box2d_center.reshape(1, 2) - gt_boxes_center) ** 2, dim=1)
			same_cls_mask = gt_clses == pred_clses[i]
			img_dis[~same_cls_mask] = 9999
			near_idx = torch.argmin(img_dis)
			# iou 2d
			iou_2d = box_iou(box2d.detach().cpu().numpy(), gt_boxes[near_idx].detach().cpu().numpy())
			
			if iou_2d < iou_thresh:
				# match failed, simply choose the default average
				continue
			else:
				estimator_index = torch.argmin(torch.abs(pred_combined_depths[i] - gt_depths[near_idx]))
				oracle_depth[i] = pred_combined_depths[i,estimator_index]
				estimated_depth_error[i] = pred_combined_uncertainty[i, estimator_index]

		return oracle_depth, estimated_depth_error

	def evaluate_3D_depths(self, targets, pred_regression):
		# computing disentangling 3D IoU for offset, depth, dimension, orientation
		batch, channel = pred_regression.shape[:2]

		# 1. extract prediction in points of interest
		target_points = targets['target_centers'].float()
		pred_regression_pois = select_point_of_interest(
			batch, target_points, pred_regression
		)

		pred_regression_pois = pred_regression_pois.view(-1, channel)
		reg_mask = targets['reg_mask'].view(-1).bool()
		pred_regression_pois = pred_regression_pois[reg_mask]
		target_points = target_points[0][reg_mask]

		# depth predictions
		pred_depths_offset = pred_regression_pois[:, self.key2channel('depth')]
		pred_keypoint_offset = pred_regression_pois[:, self.key2channel('corner_offset')]

		pred_direct_uncertainty = pred_regression_pois[:, self.key2channel('depth_uncertainty')].exp()
		pred_keypoint_uncertainty = pred_regression_pois[:, self.key2channel('corner_uncertainty')].exp()

		# dimension predictions
		target_clses = targets['cls_ids'].view(-1)[reg_mask]
		pred_dimensions_offsets = pred_regression_pois[:, self.key2channel('3d_dim')]
		pred_dimensions = self.anno_encoder.decode_dimension(
			target_clses,
			pred_dimensions_offsets,
		)
		# direct
		pred_direct_depths = self.anno_encoder.decode_depth(pred_depths_offset.squeeze(-1))
		# three depths from keypoints
		pred_keypoints_depths = self.anno_encoder.decode_depth_from_keypoints_batch(pred_keypoint_offset.view(-1, 10, 2),
																					pred_dimensions, targets['calib'],
																					)

		# combined uncertainty
		pred_combined_uncertainty = torch.cat((pred_direct_uncertainty, pred_keypoint_uncertainty), dim=1)
		pred_combined_depths = torch.cat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths), dim=1)

		# min-uncertainty
		pred_uncertainty_min_depth = pred_combined_depths[torch.arange(pred_combined_depths.shape[0]), pred_combined_uncertainty.argmin(dim=1)]
		# inv-uncertainty weighted
		# pred_uncertainty_weights = 1 / pred_combined_uncertainty**2
		pred_uncertainty_weights = 1 / pred_combined_uncertainty
		pred_uncertainty_weights = pred_uncertainty_weights / pred_uncertainty_weights.sum(dim=1, keepdim=True)
		
		pred_uncertainty_softmax_depth = torch.sum(pred_combined_depths * pred_uncertainty_weights, dim=1)

		# 3. get ground-truth
		target_locations = targets['locations'].view(-1, 3)[reg_mask]
		target_depths = target_locations[:, -1]

		# abs error
		pred_combined_error = (pred_combined_depths - target_depths[:, None]).abs()
		pred_uncertainty_min_error = (pred_uncertainty_min_depth - target_depths).abs()
		pred_uncertainty_softmax_error = (pred_uncertainty_softmax_depth - target_depths).abs()
		pred_direct_error = pred_combined_error[:, 0]
		pred_keypoints_error = pred_combined_error[:, 1:4]

		pred_mean_depth = pred_combined_depths.mean(dim=1)
		pred_mean_error = (pred_mean_depth - target_depths).abs()
		# upper-bound
		pred_min_error = pred_combined_error.min(dim=1)[0]

		pred_errors = {
						'direct': pred_direct_error,
						'direct_sigma': pred_direct_uncertainty[:, 0],
						
						'keypoint_center': pred_keypoints_error[:, 0],
						'keypoint_02': pred_keypoints_error[:, 1],
						'keypoint_13': pred_keypoints_error[:, 2],
						'keypoint_center_sigma': pred_keypoint_uncertainty[:, 0],
						'keypoint_02_sigma': pred_keypoint_uncertainty[:, 1],
						'keypoint_13_sigma': pred_keypoint_uncertainty[:, 2],

						'sigma_min': pred_uncertainty_min_error,
						'sigma_weighted': pred_uncertainty_softmax_error,
						'mean': pred_mean_error,
						'min': pred_min_error,
						'target': target_depths, 
					}

		return pred_errors

	def evaluate_3D_detection(self, targets, pred_regression):
		# computing disentangling 3D IoU for offset, depth, dimension, orientation
		batch, channel = pred_regression.shape[:2]

		# 1. extract prediction in points of interest
		target_points = targets['target_centers'].float()
		pred_regression_pois = select_point_of_interest(
			batch, target_points, pred_regression
		)

		# 2. get needed predictions
		pred_regression_pois = pred_regression_pois.view(-1, channel)
		reg_mask = targets['reg_mask'].view(-1).bool()
		pred_regression_pois = pred_regression_pois[reg_mask]
		target_points = target_points[0][reg_mask]

		pred_offset_3D = pred_regression_pois[:, self.key2channel('3d_offset')]
		pred_dimensions_offsets = pred_regression_pois[:, self.key2channel('3d_dim')]
		pred_orientation = torch.cat((pred_regression_pois[:, self.key2channel('ori_cls')], pred_regression_pois[:, self.key2channel('ori_offset')]), dim=1)
		pred_keypoint_offset = pred_regression_pois[:, self.key2channel('corner_offset')].view(-1, 10, 2)


		# 3. get ground-truth
		target_clses = targets['cls_ids'].view(-1)[reg_mask]
		target_offset_3D = targets['offset_3D'].view(-1, 2)[reg_mask]
		target_locations = targets['locations'].view(-1, 3)[reg_mask]
		target_dimensions = targets['dimensions'].view(-1, 3)[reg_mask]
		target_rotys = targets['rotys'].view(-1)[reg_mask]

		target_depths = target_locations[:, -1]

		# 4. decode prediction
		pred_dimensions = self.anno_encoder.decode_dimension(
			target_clses,
			pred_dimensions_offsets,
		)

		pred_depths_offset = pred_regression_pois[:, self.key2channel('depth')].squeeze(-1)
		if self.output_depth == 'direct':
			pred_depths = self.anno_encoder.decode_depth(pred_depths_offset)

		elif self.output_depth == 'keypoints':
			pred_depths = self.anno_encoder.decode_depth_from_keypoints(pred_offset_3D, pred_keypoint_offset, pred_dimensions, targets['calib'])
			pred_uncertainty = pred_regression_pois[:, self.key2channel('corner_uncertainty')].exp()
			pred_depths = pred_depths[torch.arange(pred_depths.shape[0]), pred_uncertainty.argmin(dim=1)]

		elif self.output_depth == 'soft':
			pred_direct_depths = self.anno_encoder.decode_depth(pred_depths_offset)
			pred_keypoints_depths = self.anno_encoder.decode_depth_from_keypoints(pred_offset_3D, pred_keypoint_offset, pred_dimensions, targets['calib'])
			pred_combined_depths = torch.cat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths), dim=1)

			pred_direct_uncertainty = pred_regression_pois[:, self.key2channel('depth_uncertainty')].exp()
			pred_keypoint_uncertainty = pred_regression_pois[:, self.key2channel('corner_uncertainty')].exp()
			pred_combined_uncertainty = torch.cat((pred_direct_uncertainty, pred_keypoint_uncertainty), dim=1)
			pred_depths = pred_combined_depths[torch.arange(pred_combined_depths.shape[0]), pred_combined_uncertainty.argmin(dim=1)]

		batch_idxs = pred_depths.new_zeros(pred_depths.shape[0]).long()
		pred_locations_offset = self.anno_encoder.decode_location_flatten(target_points, pred_offset_3D, target_depths, 
								targets['calib'], targets["pad_size"], batch_idxs)

		pred_locations_depth = self.anno_encoder.decode_location_flatten(target_points, target_offset_3D, pred_depths, 
								targets['calib'], targets["pad_size"], batch_idxs)

		pred_locations = self.anno_encoder.decode_location_flatten(target_points, pred_offset_3D, pred_depths, 
								targets['calib'], targets["pad_size"], batch_idxs)

		pred_rotys, _ = self.anno_encoder.decode_axes_orientation(
			pred_orientation,
			target_locations,
		)

		fully_pred_rotys, _ = self.anno_encoder.decode_axes_orientation(pred_orientation, pred_locations)

		# fully predicted
		pred_bboxes_3d = torch.cat((pred_locations, pred_dimensions, fully_pred_rotys[:, None]), dim=1)
		# ground-truth
		target_bboxes_3d = torch.cat((target_locations, target_dimensions, target_rotys[:, None]), dim=1)
		# disentangling
		offset_bboxes_3d = torch.cat((pred_locations_offset, target_dimensions, target_rotys[:, None]), dim=1)
		depth_bboxes_3d = torch.cat((pred_locations_depth, target_dimensions, target_rotys[:, None]), dim=1)
		dims_bboxes_3d = torch.cat((target_locations, pred_dimensions, target_rotys[:, None]), dim=1)
		orien_bboxes_3d = torch.cat((target_locations, target_dimensions, pred_rotys[:, None]), dim=1)

		# 6. compute 3D IoU
		pred_IoU = get_iou3d(pred_bboxes_3d, target_bboxes_3d)
		offset_IoU = get_iou3d(offset_bboxes_3d, target_bboxes_3d)
		depth_IoU = get_iou3d(depth_bboxes_3d, target_bboxes_3d)
		dims_IoU = get_iou3d(dims_bboxes_3d, target_bboxes_3d)
		orien_IoU = get_iou3d(orien_bboxes_3d, target_bboxes_3d)
		output = dict(pred_IoU=pred_IoU, offset_IoU=offset_IoU, depth_IoU=depth_IoU, dims_IoU=dims_IoU, orien_IoU=orien_IoU)

		return output

def get_iou3d(pred_bboxes, target_bboxes):
	num_query = target_bboxes.shape[0]

	# compute overlap along y axis
	min_h_a = - (pred_bboxes[:, 1] + pred_bboxes[:, 4] / 2)
	max_h_a = - (pred_bboxes[:, 1] - pred_bboxes[:, 4] / 2)
	min_h_b = - (target_bboxes[:, 1] + target_bboxes[:, 4] / 2)
	max_h_b = - (target_bboxes[:, 1] - target_bboxes[:, 4] / 2)

	# overlap in height
	h_max_of_min = torch.max(min_h_a, min_h_b)
	h_min_of_max = torch.min(max_h_a, max_h_b)
	h_overlap = (h_min_of_max - h_max_of_min).clamp_(min=0)

	# volumes of bboxes
	pred_volumes = pred_bboxes[:, 3] * pred_bboxes[:, 4] * pred_bboxes[:, 5]
	target_volumes = target_bboxes[:, 3] * target_bboxes[:, 4] * target_bboxes[:, 5]

	# derive x y l w alpha
	pred_bboxes = pred_bboxes[:, [0, 2, 3, 5, 6]]
	target_bboxes = target_bboxes[:, [0, 2, 3, 5, 6]]

	# convert bboxes to corners
	pred_corners = get_corners(pred_bboxes)
	target_corners = get_corners(target_bboxes)
	iou_3d = pred_bboxes.new_zeros(num_query)

	for i in range(num_query):
		ref_polygon = Polygon(pred_corners[i])
		target_polygon = Polygon(target_corners[i])
		overlap = ref_polygon.intersection(target_polygon).area
		# multiply bottom overlap and height overlap
		# for 3D IoU
		overlap3d = overlap * h_overlap[i]
		union3d = ref_polygon.area * (max_h_a[0] - min_h_a[0]) + target_polygon.area * (max_h_b[i] - min_h_b[i]) - overlap3d
		iou_3d[i] = overlap3d / union3d

	return iou_3d