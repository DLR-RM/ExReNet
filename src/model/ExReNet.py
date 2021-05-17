from tensorflow.keras.layers import Dense, Flatten,concatenate, Conv2D, Dropout, GlobalAveragePooling2D, Concatenate, UpSampling2D, BatchNormalization, Activation, LayerNormalization
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import Model as KerasModel
import numpy as np

from tensorflow.python.keras.applications.resnet import stack1

from src.utils.RMatrix import RMatrix
from src.utils.TMatrix import TMatrix
from classification_models.tfkeras import Classifiers


class ExReNet(KerasModel):
    def __init__(self, config, data):
        super(ExReNet, self).__init__()
        self.data = data
        self.config = config
        self.latent_dim = 50
        self.src_att_iters = self.config.get_list("src_att_iters")
        self.dest_att_iters = self.config.get_list("dest_att_iters")
        self.dest_att_border = self.config.get_list("dest_att_border")
        self._build()
        self.init_model()

    def init_model(self):
        # Create model parameters by doing one fake forward pass
        reference_images = tf.ones((2, 128, 128, 3))
        query_images = tf.ones((2, 128, 128, 3))
        self(reference_images, query_images, True)

    def _build_feature_network(self, load_pretrained=True):
        # Build resnet50
        feature_extractor = tf.keras.applications.ResNet50(include_top=False, weights='imagenet' if load_pretrained else None, pooling=None, input_shape=(self.data.image_size, self.data.image_size, 3))

        # Collect the layers whose outputs will be used
        emb_set_output_layers = self.config.get_list("emb_set_output_layer")
        output_layers = []
        for layer in feature_extractor.layers:
            if layer.name in emb_set_output_layers:
                output_layers.append(layer.output)

        # Declare a new model based on the new outputs
        feature_extractor = tf.keras.models.Model(feature_extractor.input, output_layers)
        output_shape = feature_extractor.output_shape
        feature_extractor.summary()

        # Do the upscaling
        if self.config.get_bool("unet_style"):
            # Go over all upscale layers
            self.upscale_layers = []
            last_channels = output_layers[-1].shape[-1]
            for filters, output_layer in zip([128, 64, 16, 4], output_layers[::-1][1:]):
                # Determine input shape
                in_shape = list(output_layer.shape)
                in_shape[-1] += last_channels
                # Build one resnet stack for that level
                inp = tf.keras.layers.Input(shape=in_shape[1:])
                x = stack1(inp, filters, 1, stride1=1, name='conv1')
                self.upscale_layers.append(tf.keras.models.Model(inp, x))
                self.upscale_layers[-1].summary()
                last_channels = x.shape[-1]

        return feature_extractor, output_shape

    def _build(self):
        # Build the feature extractor
        self.feature_extractor, feature_shape = self._build_feature_network()

        # Determine pose dimension (7+1)
        pose_dim = self.data.cam_pose_dim
        if self.config.get_bool("pred_scale_extra"):
            pose_dim += 1

        # Determine the number of channels the tensor has after the feature matching
        channels = 0
        for n in self.dest_att_iters:
            channels += 1 * n * n

        # Build regression part (resnet18)
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        resnet18 = ResNet18((128, 128, channels), include_top=False, weights=None)
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(resnet18.output)
        # Fully connected layers
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        # Dropout for MC dropout
        if self.config.get_float("dropout") > 0:
            x = Dropout(self.config.get_float("dropout"))(x, training=True)
            x = Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dense(pose_dim)(x)
        # Declare full regression part as one model
        self.decoder_pose_estimator = tf.keras.models.Model(resnet18.input, x)

    def corr_layer(self, feature_map_1, feature_map_2, last_layer_res, src_res, dest_res, upscale_facs, dest_att_border):
        batch_size = tf.shape(feature_map_1[0])[0]

        # Split up feature maps according to last corr. layer resolution
        # We gonna compute correlations for each tile individually
        feature_map_1[0] = tf.stack(tf.split(tf.stack(tf.split(feature_map_1[0], last_layer_res, 1), 1), last_layer_res, 3), 2)
        feature_map_2[0] = tf.stack(tf.split(tf.stack(tf.split(feature_map_2[0], last_layer_res, 1), 1), last_layer_res, 3), 2)

        # Collapse x,y dimensions into one dimension
        feature_map_1[0] = tf.reshape(feature_map_1[0], [batch_size, (last_layer_res) ** 2, src_res ** 2, tf.shape(feature_map_1[0])[-1]])
        feature_map_2[0] = tf.reshape(feature_map_2[0], [batch_size, (last_layer_res) ** 2, dest_res ** 2, tf.shape(feature_map_2[0])[-1]])

        # Compute dot products
        dot = tf.matmul(feature_map_1[0], feature_map_2[0], transpose_b=True)
        dot /= tf.math.sqrt(tf.cast(tf.shape(feature_map_1[0])[-1], tf.float32))

        # Find max dot products
        argmax_dot = tf.argmax(tf.reshape(dot, [-1, dest_res ** 2]), -1)
        argmax_dot = tf.reshape(argmax_dot,  tf.shape(dot)[:-1])

        # Compute feature coordinates of max dot products
        match_coord = tf.stack([argmax_dot // dest_res, argmax_dot % dest_res], -1)
        # Subtract border
        match_coord -= dest_att_border
        match_coord = tf.maximum(0, tf.minimum(dest_res - 1 - 2 * dest_att_border, tf.cast(match_coord, tf.int32)))

        # Now update the higher resolution feature maps according to the best matches
        tile_size = 1 + dest_att_border * 2
        upscale_fac = 1
        for p in range(1, len(feature_map_2)):
            # Updates tile size and upscale factor for this feature level
            tile_size *= upscale_facs[p]
            upscale_fac *= upscale_facs[p]

            # Split up feature maps according to last corr. layer resolution
            feature_map_2[p] = tf.stack(tf.split(tf.stack(tf.split(feature_map_2[p], last_layer_res, 1), 1), last_layer_res, 3), 2)

            # Collapse x,y dimensions into one dimension
            feature_map_2[p] = tf.reshape(feature_map_2[p], [batch_size, (last_layer_res) ** 2, upscale_fac * dest_res, upscale_fac * dest_res, tf.shape(feature_map_2[p])[-1]])

            # Upscale coordinates of matches to feature resolution of this level
            scaled_match_coord = tf.cast(upscale_fac * match_coord, tf.int32)

            # Build coordinates inside one tile
            base_indices = tf.cast(tf.transpose(tf.stack(tf.meshgrid(tf.range(0, tile_size),  tf.range(0, tile_size)), -1), [1, 0, 2]), tf.int32)
            # Collapse x,y dimensions into one dimension
            base_indices = tf.reshape(base_indices, [-1, 2])
            # Repeat tile coordinates for each batch
            base_indices = tf.tile(base_indices[None, None, None], tf.concat((tf.shape(argmax_dot), [1, 1]), 0))

            # Add the matched coordinates to the offset coordinates per tile, so we get full coordinates per feature vector
            base_indices += scaled_match_coord[..., None, :]
            base_indices = tf.reshape(base_indices, tf.concat((tf.shape(base_indices)[:2], [-1, 2]), 0))

            # Now reorder feature map according to the matching coordinates
            res = tf.gather_nd(feature_map_2[p], base_indices, batch_dims=2)
            res = tf.reshape(res, [batch_size, last_layer_res, last_layer_res, src_res, src_res, tile_size, tile_size, tf.shape(feature_map_2[p])[-1]])

            # Now glue one feature map together from all these tiles
            res = tf.concat(tf.unstack(res, axis=3), 4)
            res = tf.concat(tf.unstack(res, axis=3), 4)
            res = tf.concat(tf.unstack(res, axis=1), 2)
            res = tf.concat(tf.unstack(res, axis=1), 2)

            # Use the result further as new feature map
            feature_map_2[p] = res

        return dot

    def match(self, preds1, preds2):
        last_layer_res = 1
        target_size = np.prod(np.array(self.src_att_iters))
        matching = None
        all_dots = []
        matched_coordinates = []

        # Go over all correlation layers
        for i, (src_res, dest_res) in enumerate(zip(self.src_att_iters, self.dest_att_iters)):
            # Apply the corr. layer
            dot = self.corr_layer(preds1, preds2, last_layer_res, src_res, dest_res, self.src_att_iters[i:] + [128 // target_size], self.dest_att_border[i])
            # Remember dot products
            all_dots.append(dot)

            # Remember the matched last feature map which are the coordinates added via coord()
            matched_coordinates.append(preds2[-1])
            # Remove processed feature maps
            preds1 = preds1[1:]
            preds2 = preds2[1:]

            # Reshape dot products to the resolution of the feature maps
            dot = tf.reshape(dot, [tf.shape(dot)[0], last_layer_res, last_layer_res, src_res, src_res, dest_res ** 2 * 1])
            dot = tf.concat(tf.unstack(dot, axis=1), 2)
            dot = tf.concat(tf.unstack(dot, axis=1), 2)

            # Resize them to the target resolution
            dot = tf.image.resize(dot, [target_size, target_size], method='nearest')

            # Concatenate the outputs of all corr. layers
            if matching is None:
                matching = dot
            else:
                matching = tf.concat((matching, dot), -1)

            # Update last layer resolution
            last_layer_res *= src_res
        return matching, matched_coordinates, all_dots

    def coord(self, image):
        """ Creates tensor with same resolution as given image and with coordinates in cells """
        ramp = (np.arange(image.shape[-2])).astype(np.int32)
        x_ramp = np.tile(np.reshape(ramp, [1, 1, -1]), [1, image.shape[-2], 1])
        y_ramp = np.tile(np.reshape(ramp, [1, -1, 1]), [1, 1, image.shape[-2]])
        coord = tf.tile(tf.stack((x_ramp, y_ramp), -1), [tf.shape(image)[0], 1, 1, 1])
        return tf.cast(coord, tf.float32)

    def encode(self, image, training):
        # Extract features using the resnet50
        outputs = self.feature_extractor(image, training=training)
        if type(outputs) != list:
            outputs = [outputs]
        # Reverse list, so low dim features come first
        outputs = outputs[::-1]

        # Apply upscaling layers
        if self.config.get_bool("unet_style"):
            # Go over all upscaling layers
            new_outputs = [outputs[0]]
            x = outputs[0]
            for output, upscale_layer in zip(outputs[1:], self.upscale_layers):
                # Upsample by factor of 2
                x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
                # Skip connection
                x = tf.concat((x, output), axis=-1)
                # Apply resnet stack
                x = upscale_layer(x, training=training)
                new_outputs.append(x)

            # Define new output
            if len(self.config.get_list("src_att_iters")) == len(self.config.get_list("emb_set_output_layer")):
                outputs = new_outputs[:]
            elif len(self.config.get_list("src_att_iters")) == 1:
                outputs = new_outputs[-1:]
            elif len(self.config.get_list("emb_set_output_layer")) < 5:
                outputs = new_outputs[:1] + new_outputs[2:]#3] + new_outputs[4:]
            else:
                outputs = new_outputs[:1] + new_outputs[2:3] + new_outputs[4:]
        else:
            outputs = outputs[:1] + outputs[2:]

        # Add another output which has the same resolution as the input image, but contains the coordinates in each pixel
        # This is used in the matching step for computing the auxiliary loss
        coord = self.coord(image)
        outputs.append(coord)

        return outputs

    def call(self, reference_images, query_images, training, number_of_samples=1):
        # Extract features from first image
        first_features = self.encode(reference_images, training)
        # Extract features from second image
        second_features = self.encode(query_images, training)

        # Match the features
        matching, matched_coordinates, all_dots = self.match(first_features, second_features)

        # Scale up the feature matches
        matching = tf.image.resize(matching, [128, 128], 'nearest')

        # If MC dropout is activated, repeat these matches N times
        if number_of_samples > 1 and self.config.get_float("dropout") > 0:
            matching = tf.tile(matching[None], [number_of_samples, 1, 1, 1, 1])
            matching = tf.reshape(matching, tf.concat(([-1], tf.shape(matching)[2:]), 0))

        # Regress the relative pose
        cam_pose = self.decoder_pose_estimator(matching, training=training)

        return cam_pose, matched_coordinates, all_dots, matching

    @tf.function
    def _predict_using_raw_data(self, reference_images, query_image, use_uncertainty, legacy_pose_transform=False):
        # Preprocess (normalize + resize) images
        reference_images = self.data.preprocess_model_input(reference_images)
        query_image = self.data.preprocess_model_input(query_image)

        # If uncertainty estimation is enabled, predict multiple samples (MC dropout)
        N = 100 if use_uncertainty else 1

        # Forward pass
        pred, _, _, _ = self.call(reference_images, query_image, False, number_of_samples=N)

        # If scale is predicted explicitly, scale the translational direction accordingly
        if self.config.get_bool("pred_scale_extra"):
            pred = tf.concat((pred[:, :3] / tf.linalg.norm(pred[:, :3], axis=-1, keepdims=True) * pred[:, -1:], pred[:, 3:-1]), -1)

        # Post process prediction (convert to tmat)
        ref_to_query_T = self.data.postprocess_model_output(pred, legacy_pose_transform)

        # If inverse representation is predicted by model, inverse it
        if self.data.inverse_pose_representation:
            ref_to_query_T = TMatrix.inverse(ref_to_query_T, num_batch_dim=1)

        # Reshape, if multiple poses were predicted for unc. estimation
        if use_uncertainty:
            ref_to_query_T = tf.reshape(ref_to_query_T, tf.concat(([N, -1], tf.shape(ref_to_query_T)[1:]), 0))

        return ref_to_query_T

    def predict_using_raw_data(self, reference_images, query_image, use_uncertainty, legacy_pose_transform=False):
        # Regress relative pose for each reference query combination
        ref_to_query_T = []
        for i in range(len(reference_images)):
            ref_to_query_T.append(self._predict_using_raw_data(reference_images[i:i+1], query_image[None], use_uncertainty, legacy_pose_transform))
        ref_to_query_T = tf.concat(ref_to_query_T, 1 if use_uncertainty else 0)

        # Optional: Estimate uncertainty based on mean deviation
        if use_uncertainty:
            ref_to_query_T = TMatrix.to_quaternion(ref_to_query_T, 2)
            # Compute mean camera pose
            mean_pose = tf.reduce_mean(ref_to_query_T[..., :3], 0)
            # Compute mean deviation from mean and use as unc.
            uncertainty = tf.reduce_mean(tf.linalg.norm(ref_to_query_T[..., :3] - mean_pose, axis=-1), 0)

            # Compute mean of relative rotation estimates
            mean_quats = []
            for i in range(ref_to_query_T.shape[1]):
                mean_quats.append(RMatrix.average_quaternions(ref_to_query_T[:, i, ..., 3:]))
            mean_quats = tf.cast(tf.stack(mean_quats, 0), tf.float32)

            # Rebuild relative transformation mat
            ref_to_query_T = TMatrix.from_quaternion(tf.concat((mean_pose, mean_quats), -1))

        # Form nice dicts for each relative pose estimation (here R1=R2)
        ref_to_query = []
        for i in range(len(ref_to_query_T)):
            ref_to_query.append({
                "t": np.stack([ref_to_query_T[i, 0, 3], ref_to_query_T[i, 1, 3], ref_to_query_T[i, 2, 3]], -1),
                "R1": np.array(ref_to_query_T[i, :3, :3]),
                "R2": np.array(ref_to_query_T[i, :3, :3])
            })

        if use_uncertainty:
            return ref_to_query, uncertainty
        else:
            return ref_to_query, None
