import random

import mathutils

from src.main.Provider import Provider
import numpy as np


class Mixed3d(Provider):
    """ Samples a 3-dimensional vector based on clipped mixed distributions.

    **Configuration**:

    .. list-table::
        :widths: 25 100 10
        :header-rows: 1

        * - Parameter
          - Description
          - Type
        * - min
          - Lower limit of allowed values. If one of the sampled values is below that, it is resampled.
          - list
        * - max
          - Upper limit of allowed values. If one of the sampled values is above that, it is resampled.
          - list
        * - param1
          - Means of gaussian distributions / Min of uniform distribution
          - list
        * - param2
          - Standard deviation of gaussian distributions / Max of uniform distribution
          - list
        * - distr
          - Setting the distribution type per coordinate. Allowed values: ["gaussian", "uniform"]
          - list
        * - convert_deg_to_rad
          - If values are in degrees and should be converted to radians.
          - bool
    """

    def __init__(self, config):
        Provider.__init__(self, config)

    def run(self):
        """
        :return: Sampled value. Type: Mathutils Vector
        """
        min = self.config.get_vector3d("min")
        max = self.config.get_vector3d("max")
        param1 = self.config.get_vector3d("param1")
        param2 = self.config.get_vector3d("param2")
        distr = self.config.get_list("distr")
        convert_deg_to_rad = self.config.get_bool("convert_deg_to_rad", False)

        position = mathutils.Vector()
        for i in range(3):
            val = None
            while val is None:
                if distr[i] == "gaussian":
                    val = random.normalvariate(param1[i], param2[i])
                elif distr[i] == "uniform":
                    val = random.uniform(param1[i], param2[i])
                else:
                    raise Exception("No such distribution: " + distr[i])

                if val < min[i] or val > max[i]:
                    val = None

            position[i] = (val / 180 * np.pi) if convert_deg_to_rad else val

        return position